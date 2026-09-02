"""Scripted :class:`LLMClient` for tests — no network, fully deterministic.

This is the payoff of keeping :class:`~jpm_ai_advisor.llm.base.LLMClient`
small: the orchestrator and all three agents can be exercised end to end in
``pytest`` by handing each one a queue of canned responses, with no mocking
framework and no live API calls.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .base import LLMResponse, Message, ToolSchema


@dataclass
class FakeLLMClient:
    responses: deque[LLMResponse]

    def __init__(self, responses: list[LLMResponse]) -> None:
        self.responses = deque(responses)
        self.calls: list[dict] = []

    def generate(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[ToolSchema],
        enable_web_search: bool = False,
    ) -> LLMResponse:
        self.calls.append({"system": system, "messages": list(messages), "tools": list(tools)})
        if not self.responses:
            raise AssertionError("FakeLLMClient ran out of scripted responses")
        return self.responses.popleft()
