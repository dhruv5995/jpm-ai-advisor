from __future__ import annotations

import json

from ..llm.base import LLMClient
from ..protocol import ClientReply
from .base import Agent, ToolSpec

SYSTEM_PROMPT_TEMPLATE = """You are roleplaying a financial advisory client with this profile:

{profile_json}

Stay fully in character as this specific person talking to their financial advisor.

Rules:
- Speak in first person, plain language a non-expert would use — no financial jargon you
  wouldn't plausibly know yourself.
- Ground everything you say in your specific numbers, goals, and constraints above. Don't
  invent new facts about yourself that aren't in the profile.
- Be reasonably skeptical: if the advisor's answer is vague, generic, or ignores your specific
  situation, push back or ask a concrete follow-up instead of accepting it.
- Don't stall indefinitely — once the advisor gives you a specific, actionable recommendation
  that actually addresses what you asked about, set satisfied=True. Don't hold out for
  perfection.
- You must respond by calling the `reply` tool exactly once per turn.
"""


class ClientAgent(Agent):
    def __init__(self, *, llm: LLMClient, profile: dict) -> None:
        self.profile = profile
        super().__init__(
            name="client",
            system_prompt=SYSTEM_PROMPT_TEMPLATE.format(profile_json=json.dumps(profile, indent=2)),
            llm=llm,
            tools=[ToolSpec(name="reply", model=ClientReply, description="Send your reply to the advisor.")],
        )

    def open_conversation(self) -> ClientReply:
        self.receive(
            "Start the conversation: bring up whatever is most on your mind about your finances "
            "right now, in your own words, as if this were the start of a real meeting with your "
            "advisor."
        )
        _, reply = self.step()
        assert isinstance(reply, ClientReply)
        return reply
