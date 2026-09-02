"""Minimal, provider-agnostic chat + tool-calling contract.

This is deliberately not LangChain's ``BaseChatModel``: we need exactly three
things (send a system prompt + turn history + tool schemas, get back text
and/or a tool call, and be able to swap in a scripted fake for tests) — not a
general-purpose abstraction over every provider feature. Keeping this small
is itself a design choice: a thin ``Protocol`` plus a handful of frozen
dataclasses is enough to add a second provider or a test double without
touching agent code, and it stays trivial to read top to bottom.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class ToolSchema:
    """A tool the model may call, described as JSON schema."""

    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True)
class ToolCall:
    """A model-issued request to invoke one of our custom tools."""

    id: str
    name: str
    input: dict[str, Any]


@dataclass(frozen=True)
class ToolResult:
    """The result we feed back for a previously issued ToolCall."""

    tool_call_id: str
    tool_name: str
    content: str


@dataclass(frozen=True)
class Citation:
    """A source surfaced by a hosted (server-executed) tool, e.g. web search."""

    url: str
    title: str = ""


@dataclass(frozen=True)
class Message:
    """One turn in a conversation, in our own provider-neutral shape."""

    role: str  # "user" | "assistant"
    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    tool_results: tuple[ToolResult, ...] = ()


@dataclass(frozen=True)
class LLMResponse:
    """What a provider adapter hands back for one ``generate`` call."""

    text: str
    tool_calls: tuple[ToolCall, ...] = ()
    citations: tuple[Citation, ...] = field(default_factory=tuple)


class LLMClient(Protocol):
    """Provider adapters implement this; agents only ever depend on this."""

    def generate(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[ToolSchema],
        enable_web_search: bool = False,
    ) -> LLMResponse:
        """Run one model turn.

        ``tools`` are custom, client-executed tools (we see a ``ToolCall``
        and must execute it and append a ``ToolResult`` ourselves).
        ``enable_web_search`` turns on the provider's *hosted* search tool,
        which the provider executes server-side inline within this same
        call — its results come back already folded into ``text`` and
        ``citations``, never as a ``ToolCall`` we need to handle.
        """
        ...
