"""Turn-taking loop that drives the three agents toward a resolution.

This is the piece a multi-agent framework would normally give you as a graph
executor or a group-chat manager. Written out here, it's ~50 lines: seed the
client's opening message, let the advisor decide (possibly bouncing off the
analyst any number of times) whether to research more or respond, get the
client's reaction, repeat until the client is satisfied or the advisor
explicitly ends the session. Two safety valves bound it: a max number of
analyst delegations per advisor turn, and a max number of client turns per
session — without them, a model that never converges would loop forever.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .agents.advisor_agent import AdvisorAgent
from .agents.analyst_agent import AnalystAgent
from .agents.client_agent import ClientAgent
from .llm.base import LLMClient
from .protocol import AskClient, DelegateToAnalyst, EndConversation, ReportToAdvisor
from .tools.web_search import format_citations

MAX_DELEGATIONS_PER_TURN = 5


@dataclass(frozen=True)
class TranscriptEntry:
    speaker: str  # "client" | "advisor" | "analyst" | "system"
    channel: str  # "client<->advisor" | "advisor<->analyst" | "system"
    text: str


@dataclass
class SessionResult:
    resolved: bool
    summary: str
    transcript: list[TranscriptEntry] = field(default_factory=list)

    def to_markdown(self) -> str:
        lines = [
            "# Advisory session transcript",
            "",
            f"**Resolution:** {'resolved' if self.resolved else 'unresolved'}",
            f"**Summary:** {self.summary}",
            "",
        ]
        for entry in self.transcript:
            lines.append(f"### {entry.speaker} ({entry.channel})")
            lines.append(entry.text)
            lines.append("")
        return "\n".join(lines)


def run_session(*, llm: LLMClient, profile: dict, max_client_turns: int = 8) -> SessionResult:
    client = ClientAgent(llm=llm, profile=profile)
    advisor = AdvisorAgent(llm=llm)
    analyst = AnalystAgent(llm=llm)

    transcript: list[TranscriptEntry] = []

    def log(speaker: str, channel: str, text: str) -> None:
        transcript.append(TranscriptEntry(speaker=speaker, channel=channel, text=text))

    opening = client.open_conversation()
    log("client", "client<->advisor", opening.message)
    advisor.receive(f"Client says: {opening.message}")

    for _ in range(max_client_turns):
        for _ in range(MAX_DELEGATIONS_PER_TURN):
            _, action = advisor.step()

            if isinstance(action, DelegateToAnalyst):
                log("advisor", "advisor<->analyst", f"Task: {action.task}\nContext: {action.context}")
                analyst.receive(f"Task: {action.task}\nContext: {action.context}")
                _, report = analyst.step()
                assert isinstance(report, ReportToAdvisor)

                report_text = report.findings
                if report.sources:
                    report_text += "\nKnowledge base sources: " + ", ".join(report.sources)
                web_sources = format_citations(analyst.last_citations)
                if web_sources:
                    report_text += "\n" + web_sources

                log("analyst", "advisor<->analyst", report_text)
                advisor.receive(f"Analyst reports: {report_text}")
                continue

            if isinstance(action, AskClient):
                log("advisor", "client<->advisor", action.message)
                client.receive(f"Advisor says: {action.message}")
                break

            if isinstance(action, EndConversation):
                log("advisor", "system", f"[end_conversation] {action.summary}")
                return SessionResult(action.resolution == "resolved", action.summary, transcript)
        else:
            return SessionResult(False, "Advisor exceeded delegation budget without a response.", transcript)

        _, reply = client.step()
        log("client", "client<->advisor", reply.message)
        if reply.satisfied:
            return SessionResult(True, reply.message, transcript)
        advisor.receive(f"Client says: {reply.message}")

    return SessionResult(False, "Max client turns reached without resolution.", transcript)
