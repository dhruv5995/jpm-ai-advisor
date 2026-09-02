from __future__ import annotations

from ..llm.base import LLMClient
from ..protocol import KnowledgeSearch, ReportToAdvisor
from ..tools.knowledge_search import KnowledgeStore
from .base import Agent, ToolSpec

SYSTEM_PROMPT = """You are the analyst supporting a financial advisor. You never speak to the
client directly — you only receive research tasks from the advisor and report findings back.

You have two research tools:
- `knowledge_search`: search the firm's internal knowledge base (asset class data, model
  portfolios, glossary, firm principles/disclosures).
- `web_search`: search the live internet for current information (rates, prices, recent news).

Rules:
- Ground every factual claim in what your tools actually returned. Never state a specific
  number, rate, or fact you have not just retrieved.
- Use knowledge_search first for firm model portfolios, asset-class characteristics, or
  definitions. Use web_search for anything that changes over time (current rates, prices,
  recent news) or isn't in the internal knowledge base.
- When you have enough to answer the advisor's task, call `report_to_advisor` with a concise
  findings summary and the sources you used. Don't keep researching past what the task needs.
"""


class AnalystAgent(Agent):
    def __init__(self, *, llm: LLMClient, knowledge_store: KnowledgeStore | None = None) -> None:
        self.knowledge_store = knowledge_store or KnowledgeStore()
        super().__init__(
            name="analyst",
            system_prompt=SYSTEM_PROMPT,
            llm=llm,
            enable_web_search=True,
            # Higher than the default: with parallel tool calls disabled
            # (protocol.py requires one decision per turn), a multi-part
            # research task now takes one knowledge_search per turn instead
            # of batching several at once, so it needs more turns to reach
            # report_to_advisor.
            max_internal_steps=10,
            tools=[
                ToolSpec(
                    name="knowledge_search",
                    model=KnowledgeSearch,
                    description="Search the internal knowledge base.",
                    executor=lambda args: self.knowledge_store.search_as_text(args.query),
                ),
                ToolSpec(
                    name="report_to_advisor",
                    model=ReportToAdvisor,
                    description="Report your findings back to the advisor.",
                ),
            ],
        )
