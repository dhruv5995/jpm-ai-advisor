from __future__ import annotations

from ..llm.base import LLMClient
from ..protocol import AskClient, DelegateToAnalyst, EndConversation
from .base import Agent, ToolSpec

SYSTEM_PROMPT = """You are a financial advisor. You are the ONLY agent who speaks with both the
client and the analyst — the client and analyst never interact directly, so you are
responsible for carrying context between them.

Rules:
- Never state a specific fact, number, rate, or product recommendation to the client unless it
  came from an analyst report earlier in this conversation. If you don't have what you need
  yet, delegate to the analyst instead of guessing.
- Keep analyst tasks specific and answerable ("current model portfolio for risk band 2", not
  "tell me about bonds"). Include whatever client context — age, risk aversion, goal, amounts
  — the analyst needs to make the research relevant.
- Speak to the client in plain, warm, professional language. Translate analyst findings into
  advice; don't paste raw research at them.
- You can only ever conclude the session successfully by sending the client your recommendation
  via `ask_client` and having them confirm it addresses their need — never by calling
  `end_conversation` instead of replying. If the client asks follow-up questions, answer them
  (delegating to the analyst again if needed) rather than declaring the session done.
- `end_conversation` is a give-up path only, for when the conversation is going in circles
  without productive progress. It always ends the session unresolved — it is never a shortcut
  for "the client's need is addressed," which must instead be reached by continuing to talk to
  the client until they are satisfied.
- Every turn, call exactly one of: ask_client, delegate_to_analyst, end_conversation.
"""


class AdvisorAgent(Agent):
    def __init__(self, *, llm: LLMClient) -> None:
        super().__init__(
            name="advisor",
            system_prompt=SYSTEM_PROMPT,
            llm=llm,
            tools=[
                ToolSpec(name="ask_client", model=AskClient, description="Send a message to the client."),
                ToolSpec(
                    name="delegate_to_analyst",
                    model=DelegateToAnalyst,
                    description="Hand a research task to the analyst.",
                ),
                ToolSpec(
                    name="end_conversation",
                    model=EndConversation,
                    description="Give up on this session without resolution (not for a completed answer).",
                ),
            ],
        )
