"""Shared inter-agent contract: one pydantic model per allowed action.

Why structured tool calls instead of parsing free text ("if 'I need to
research' in reply..."): every multi-agent framework (LangGraph, AutoGen,
CrewAI) ultimately routes on *something* structured — a graph edge, a
function call, a typed handoff object. Free-text routing is what you get if
you skip that layer, and it's brittle: it breaks the moment a model phrases
itself differently. We make the same choice those frameworks make, just
explicitly: each agent's turn must end in exactly one of a small, closed set
of pydantic-validated actions, so routing is a dict lookup on a name, not a
string match on prose.

Using pydantic (rather than hand-written JSON schema dicts) buys two things:
``model_json_schema()`` gives the provider-facing tool schema for free, and
``model_validate()`` gives us a validated, typed object back from whatever
dict the model produced — with a clear ``ValidationError`` if the model
hands back something malformed, instead of a silent ``KeyError`` three
frames later.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from .llm.base import ToolSchema

# --- Advisor actions -------------------------------------------------------


class AskClient(BaseModel):
    """Advisor sends a message directly to the client and waits for a reply."""

    message: str = Field(description="The message to send to the client, in plain advisory language.")


class DelegateToAnalyst(BaseModel):
    """Advisor hands a research task to the analyst."""

    task: str = Field(description="A specific, answerable research question for the analyst.")
    context: str = Field(default="", description="Relevant client context the analyst needs.")


class EndConversation(BaseModel):
    """Advisor gives up on the session because it isn't converging.

    There is deliberately no "resolved" option here. A live run against
    GPT-5.1 showed this tool being used to unilaterally declare victory —
    calling it right after receiving analyst findings, without ever sending
    the client anything via `ask_client`, once even skipping two follow-up
    questions the client had just asked. Only the client's own
    ``ClientReply.satisfied`` can mark a session resolved; this tool exists
    solely for the "we're going in circles, stop" case.
    """

    summary: str = Field(description="A short summary of why the session is being ended without resolution.")


# --- Analyst actions ---------------------------------------------------------


class KnowledgeSearch(BaseModel):
    """Analyst queries the internal knowledge base."""

    query: str = Field(description="A short search query over the internal knowledge base.")


class ReportToAdvisor(BaseModel):
    """Analyst hands findings back to the advisor."""

    findings: str = Field(description="A concise, factual summary of what was found.")
    sources: list[str] = Field(default_factory=list, description="URLs or knowledge-base doc ids used.")


# --- Client actions ----------------------------------------------------------


class ClientReply(BaseModel):
    """Client's response to the advisor."""

    message: str = Field(description="The client's reply, in first person, matching their persona.")
    satisfied: bool = Field(
        description="True only if this reply fully resolves the client's current need with no more questions."
    )


def to_tool_schema(model: type[BaseModel], name: str, description: str) -> ToolSchema:
    schema = model.model_json_schema()
    schema.pop("title", None)
    return ToolSchema(name=name, description=description, input_schema=schema)
