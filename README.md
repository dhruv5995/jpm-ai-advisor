# jpm-ai-advisor

A multi-agent simulated financial advisory conversation: a **Client** agent
(roleplaying a dummy persona), an **Advisor** agent (the only one who talks
to the client, and to the analyst), and an **Analyst** agent (internet + a
local knowledge base, never talks to the client) — talking to each other
until the client's need is resolved or a turn budget is hit.

## Quickstart

```bash
uv sync
cp .env.example .env   # fill in ANTHROPIC_API_KEY or OPENAI_API_KEY
uv run python -m jpm_ai_advisor.main --profile conservative
```

`--profile` is `conservative` (Diane, 58, near retirement) or `growth`
(Marcus, 29, saving for a house *and* retirement — two goals with different
time horizons on purpose). `--provider anthropic|openai` overrides
`LLM_PROVIDER` from `.env`. Each run prints the transcript and saves it to
`transcripts/`.

```bash
uv run pytest      # no network required — the LLM is a scripted fake
uv run ruff check .
```

## Architecture

```
Client  <---->  Advisor  <---->  Analyst
                                     |
                              web_search (hosted)
                              knowledge_search (local BM25)
```

- **Structured tool-calling, not free text, drives routing.** Every agent
  turn ends in exactly one schema-validated action (`protocol.py`):
  Advisor → `ask_client` / `delegate_to_analyst` / `end_conversation`;
  Analyst → `knowledge_search` / `web_search` / `report_to_advisor`;
  Client → `reply(message, satisfied)`. Termination is `satisfied=True` or
  an explicit `end_conversation`, not string-matching on prose.
- **No multi-agent framework.** The orchestration loop
  (`orchestrator.py`) is hand-written — about 50 lines — specifically so
  every routing/handoff/termination decision is visible and explainable
  rather than living inside LangGraph/AutoGen/CrewAI internals.
- **Provider-agnostic core.** `llm/base.py` defines a minimal `LLMClient`
  Protocol; `llm/anthropic_client.py` and `llm/openai_client.py` are thin
  adapters, and `llm/fake_client.py` is a scripted double used in tests.
- **Context isolation is explicit.** Each agent has its own message
  history; the client never sees the advisor↔analyst channel.

Full first-principles rationale, prompting tradeoffs, and what a framework
would have done differently live in a separate (not-committed) notes doc
kept alongside this repo for interview prep.

## Layout

```
src/jpm_ai_advisor/
  llm/            provider-agnostic client + Anthropic/OpenAI/fake adapters
  protocol.py     pydantic tool-call schemas (the inter-agent contract)
  agents/         Agent base loop + Client/Advisor/Analyst
  tools/          knowledge_search (BM25) + web_search (hosted-tool notes)
  knowledge_base/ curated markdown docs the analyst can retrieve from
  profiles/       dummy client personas
  orchestrator.py turn-taking loop + transcript
  main.py         CLI
tests/            pytest, all against FakeLLMClient — no API key needed
```
