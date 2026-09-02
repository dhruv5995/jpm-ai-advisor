"""Anthropic adapter for :class:`~jpm_ai_advisor.llm.base.LLMClient`."""

from __future__ import annotations

from anthropic import Anthropic

from .base import Citation, LLMResponse, Message, ToolCall, ToolSchema

DEFAULT_MODEL = "claude-sonnet-5"


class AnthropicClient:
    def __init__(self, api_key: str, model: str = DEFAULT_MODEL) -> None:
        self._client = Anthropic(api_key=api_key)
        self._model = model

    def generate(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[ToolSchema],
        enable_web_search: bool = False,
    ) -> LLMResponse:
        api_tools: list[dict] = [
            {"name": t.name, "description": t.description, "input_schema": t.input_schema}
            for t in tools
        ]
        if enable_web_search:
            api_tools.append({"type": "web_search_20250305", "name": "web_search", "max_uses": 3})

        response = self._client.messages.create(
            model=self._model,
            # Sized generously on purpose: this model thinks by default, and
            # thinking tokens count against max_tokens before the tool-call
            # JSON is written. A budget sized for "just the structured
            # output" (e.g. 2048) starts truncating tool calls mid-JSON on
            # longer research turns — silently producing a `{}` argument
            # set that then fails validation. Found by an actual live run,
            # not anticipated up front.
            max_tokens=8192,
            system=system,
            messages=[_to_api_message(m) for m in messages],
            tools=api_tools,
            # Our protocol (protocol.py) requires exactly one decision tool
            # call per turn; without this, the model will happily emit two
            # (e.g. two `delegate_to_analyst` calls at once), and only one
            # can become "the" decision the orchestrator acts on — silently
            # dropping the other. Enforcing it here, once, beats trying to
            # reconcile multiple simultaneous decisions in orchestration
            # logic.
            tool_choice={"type": "auto", "disable_parallel_tool_use": True},
        )

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        citations: list[Citation] = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
                for c in getattr(block, "citations", None) or []:
                    if getattr(c, "type", "") == "web_search_result_location":
                        citations.append(Citation(url=c.url, title=getattr(c, "title", "") or ""))
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(id=block.id, name=block.name, input=dict(block.input)))
            # server_tool_use / web_search_tool_result blocks are the
            # provider's own search steps: already resolved server-side, we
            # only need the citations attached to the final text block.

        return LLMResponse(text="".join(text_parts), tool_calls=tuple(tool_calls), citations=tuple(citations))


def _to_api_message(message: Message) -> dict:
    content: list[dict] = []
    if message.text:
        content.append({"type": "text", "text": message.text})
    for call in message.tool_calls:
        content.append({"type": "tool_use", "id": call.id, "name": call.name, "input": call.input})
    for result in message.tool_results:
        content.append(
            {"type": "tool_result", "tool_use_id": result.tool_call_id, "content": result.content}
        )
    return {"role": message.role, "content": content}
