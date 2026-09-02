"""Generic tool-calling agent loop shared by all three agents.

An agent's turn is: call the model, and if it called a *working* tool (one
we execute ourselves, e.g. knowledge search) feed the result back and call
again; if it called a *decision* tool (one of the small closed set from
``protocol.py`` with no executor attached), stop and hand that decision back
to the orchestrator. ``max_internal_steps`` is a hard cap so a model that
never converges on a decision fails loudly instead of looping forever and
silently burning API calls — a scenario a framework's built-in retry/step
budget would normally hide from you.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from pydantic import BaseModel, ValidationError

from ..llm.base import Citation, LLMClient, Message, ToolResult
from ..protocol import to_tool_schema


@dataclass(frozen=True)
class ToolSpec:
    name: str
    model: type[BaseModel]
    description: str
    executor: Callable[[BaseModel], str] | None = None
    """None marks this as a terminal *decision* tool: calling it ends the
    agent's turn instead of being auto-executed and looped back in."""


class Agent:
    def __init__(
        self,
        *,
        name: str,
        system_prompt: str,
        llm: LLMClient,
        tools: list[ToolSpec],
        enable_web_search: bool = False,
        max_internal_steps: int = 5,
    ) -> None:
        self.name = name
        self.system_prompt = system_prompt
        self.llm = llm
        self.tools = {t.name: t for t in tools}
        self.enable_web_search = enable_web_search
        self.max_internal_steps = max_internal_steps
        self.history: list[Message] = []
        self.last_citations: tuple[Citation, ...] = ()

    def receive(self, text: str) -> None:
        self.history.append(Message(role="user", text=text))

    def step(self) -> tuple[str, BaseModel]:
        """Run until a decision tool is called; return (tool_name, parsed_args)."""
        schemas = [to_tool_schema(t.model, t.name, t.description) for t in self.tools.values()]
        citations: list[Citation] = []

        for _ in range(self.max_internal_steps):
            response = self.llm.generate(
                system=self.system_prompt,
                messages=self.history,
                tools=schemas,
                enable_web_search=self.enable_web_search,
            )
            citations.extend(response.citations)
            self.history.append(Message(role="assistant", text=response.text, tool_calls=response.tool_calls))

            if not response.tool_calls:
                nudge = "You must respond by calling exactly one of your available tools."
                self.history.append(Message(role="user", text=nudge))
                continue

            tool_results: list[ToolResult] = []
            terminal: tuple[str, BaseModel] | None = None
            for call in response.tool_calls:
                spec = self.tools.get(call.name)
                if spec is None:
                    unknown_tool = f"Unknown tool: {call.name}"
                    tool_results.append(
                        ToolResult(tool_call_id=call.id, tool_name=call.name, content=unknown_tool)
                    )
                    continue

                try:
                    parsed = spec.model.model_validate(call.input)
                except ValidationError as exc:
                    # A malformed tool call is exactly what pydantic
                    # validation is for catching, but validating is only
                    # half the benefit — feeding the error back as a
                    # tool_result lets the model see and correct its own
                    # mistake next turn instead of crashing the whole
                    # session on a schema slip (e.g. `sources` returned as
                    # a comma-joined string instead of a JSON list).
                    error_text = f"Invalid arguments: {exc}"
                    tool_results.append(
                        ToolResult(tool_call_id=call.id, tool_name=call.name, content=error_text)
                    )
                    continue

                if spec.executor is None:
                    # A decision tool is terminal for *our* orchestration,
                    # but the provider still requires every tool_use in this
                    # message to get a paired tool_result before the
                    # conversation continues (Anthropic rejects the next
                    # call otherwise: "tool_use ids were found without
                    # tool_result blocks immediately after"). Acknowledge it
                    # like any other, and only return once every call in
                    # this response — including any that came before it —
                    # has been closed out below.
                    if terminal is None:
                        terminal = (call.name, parsed)
                    tool_results.append(
                        ToolResult(tool_call_id=call.id, tool_name=call.name, content="Acknowledged.")
                    )
                    continue

                result = spec.executor(parsed)
                tool_results.append(ToolResult(tool_call_id=call.id, tool_name=call.name, content=result))

            self.history.append(Message(role="user", tool_results=tuple(tool_results)))

            if terminal is not None:
                self.last_citations = tuple(citations)
                return terminal

        raise RuntimeError(
            f"{self.name} exceeded max_internal_steps ({self.max_internal_steps}) without a decision"
        )
