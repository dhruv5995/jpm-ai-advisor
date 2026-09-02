"""End-to-end orchestration test against a scripted LLM — no network calls.

The point of the thin ``LLMClient`` protocol: this test exercises the real
``ClientAgent``/``AdvisorAgent``/``AnalystAgent``/``run_session`` code, not
mocks of it, by handing them a single ``FakeLLMClient`` pre-loaded with the
exact sequence of tool calls a real model would make for one clean
resolution path (research once, then answer, then the client is satisfied).
"""

from jpm_ai_advisor.llm.base import LLMResponse, ToolCall
from jpm_ai_advisor.llm.fake_client import FakeLLMClient
from jpm_ai_advisor.orchestrator import run_session

PROFILE = {
    "name": "Test Client",
    "age": 40,
    "risk_aversion": 3,
    "goals": ["Retire comfortably"],
}


def _tool_response(name: str, **input_) -> LLMResponse:
    return LLMResponse(text="", tool_calls=(ToolCall(id=f"call-{name}", name=name, input=input_),))


def test_run_session_resolves_via_one_research_round_trip() -> None:
    llm = FakeLLMClient(
        [
            # 1. client opens the conversation
            _tool_response("reply", message="What should my portfolio look like?", satisfied=False),
            # 2. advisor delegates research
            _tool_response(
                "delegate_to_analyst",
                task="Recommend a model portfolio for risk band 3",
                context="40 years old, moderate risk aversion, retirement goal",
            ),
            # 3. analyst searches the knowledge base
            _tool_response("knowledge_search", query="risk band 3 model portfolio"),
            # 4. analyst reports back
            _tool_response(
                "report_to_advisor",
                findings="Risk band 3 (Balanced): 40% US large-cap, 15% intl, 5% EM, 35% bonds, 5% cash.",
                sources=["model_portfolios#Risk Band 3 — Balanced"],
            ),
            # 5. advisor answers the client
            _tool_response(
                "ask_client",
                message="Based on your profile, a balanced mix of 40% US stocks, 15% international, "
                "5% emerging markets, 35% bonds, and 5% cash fits your goals.",
            ),
            # 6. client is satisfied
            _tool_response("reply", message="That makes sense, thank you.", satisfied=True),
        ]
    )

    result = run_session(llm=llm, profile=PROFILE, max_client_turns=8)

    assert result.resolved is True
    speakers = [entry.speaker for entry in result.transcript]
    assert speakers == ["client", "advisor", "analyst", "advisor", "client"]
    assert "Balanced" in result.transcript[2].text


def test_run_session_gives_up_after_max_client_turns() -> None:
    responses = [_tool_response("reply", message="opening", satisfied=False)]
    for _ in range(8):
        responses.append(_tool_response("ask_client", message="still working on it"))
        responses.append(_tool_response("reply", message="not yet", satisfied=False))
    llm = FakeLLMClient(responses)

    result = run_session(llm=llm, profile=PROFILE, max_client_turns=8)

    assert result.resolved is False
    assert "Max client turns" in result.summary
