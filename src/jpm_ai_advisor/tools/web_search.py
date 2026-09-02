"""Internet access for the analyst agent.

There's no client-executed search function here on purpose. Both providers
expose web search as a *hosted* tool (Anthropic's ``web_search_20250305``,
OpenAI's ``web_search`` on the Responses API): the provider runs the search
and fetches pages server-side, inside the same API call, and hands back
already-grounded text plus citations. Standing up our own client-executed
search tool (Tavily/SerpAPI/Bing + an HTTP client + result parsing) would
duplicate work the provider already does, for no upside here — and it would
need its own API key, which a hosted tool avoids entirely.

The trade-off, worth naming explicitly: we're coupled to whatever the
provider's search tool decides to fetch and how it summarizes it — we can't
inspect intermediate results or swap search backends. A production system
that needed that control (an approved-sources allowlist that goes beyond a
`allowed_domains` filter, custom re-ranking, audit logging of raw fetched
pages) would need to fall back to a client-executed tool with a real search
API behind it, at the cost of an extra dependency and an extra key.

``AnalystAgent`` enables the hosted tool via ``enable_web_search=True`` on
its :class:`~jpm_ai_advisor.agents.base.Agent`; the provider adapters in
``llm/`` translate that into each API's native hosted-tool shape.
"""

from __future__ import annotations

from ..llm.base import Citation


def format_citations(citations: tuple[Citation, ...]) -> str:
    if not citations:
        return ""
    lines = [f"- {c.title or c.url} ({c.url})" for c in citations]
    return "Sources:\n" + "\n".join(lines)
