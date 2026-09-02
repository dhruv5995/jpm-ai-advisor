"""OpenAI adapter for :class:`~jpm_ai_advisor.llm.base.LLMClient`.

Built on the Responses API rather than Chat Completions specifically because
Responses is the one place OpenAI exposes a *hosted* web-search tool
alongside ordinary function tools in a single unified ``tools`` list — the
same shape Anthropic's Messages API gives us. Chat Completions would need a
separate search API entirely, breaking the provider-agnostic story.
"""

from __future__ import annotations

import json

from openai import OpenAI

from .base import Citation, LLMResponse, Message, ToolCall, ToolSchema

DEFAULT_MODEL = "gpt-5.1"


class OpenAIClient:
    def __init__(self, api_key: str, model: str = DEFAULT_MODEL) -> None:
        self._client = OpenAI(api_key=api_key)
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
            {
                "type": "function",
                "name": t.name,
                "description": t.description,
                "parameters": t.input_schema,
            }
            for t in tools
        ]
        if enable_web_search:
            api_tools.append({"type": "web_search"})

        response = self._client.responses.create(
            model=self._model,
            instructions=system,
            input=_to_api_input(messages),
            tools=api_tools,
            # See the matching comment in anthropic_client.py: our protocol
            # requires exactly one decision tool call per turn.
            parallel_tool_calls=False,
        )

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        citations: list[Citation] = []
        for item in response.output:
            if item.type == "message":
                for part in item.content:
                    if getattr(part, "type", "") == "output_text":
                        text_parts.append(part.text)
                        for a in getattr(part, "annotations", None) or []:
                            if getattr(a, "type", "") == "url_citation":
                                citations.append(Citation(url=a.url, title=getattr(a, "title", "") or ""))
            elif item.type == "function_call":
                tool_calls.append(
                    ToolCall(id=item.call_id, name=item.name, input=json.loads(item.arguments or "{}"))
                )
            # "web_search_call" items are the provider's own search steps,
            # already resolved server-side; nothing for us to execute.

        return LLMResponse(text="".join(text_parts), tool_calls=tuple(tool_calls), citations=tuple(citations))


def _to_api_input(messages: list[Message]) -> list[dict]:
    items: list[dict] = []
    for message in messages:
        if message.text:
            items.append({"role": message.role, "content": message.text})
        for call in message.tool_calls:
            items.append(
                {
                    "type": "function_call",
                    "call_id": call.id,
                    "name": call.name,
                    "arguments": json.dumps(call.input),
                }
            )
        for result in message.tool_results:
            items.append(
                {"type": "function_call_output", "call_id": result.tool_call_id, "output": result.content}
            )
    return items
