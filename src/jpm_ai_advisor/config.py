"""Environment-driven provider selection.

One function, one job: read env vars, fail loudly if the chosen provider's
key is missing, and hand back an :class:`~jpm_ai_advisor.llm.base.LLMClient`.
Agents and the orchestrator never import a provider adapter or touch
``os.environ`` directly — this is the single seam where that happens, which
is what actually makes the provider swappable rather than just theoretically
swappable.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

from .llm.anthropic_client import DEFAULT_MODEL as ANTHROPIC_DEFAULT_MODEL
from .llm.anthropic_client import AnthropicClient
from .llm.base import LLMClient
from .llm.openai_client import DEFAULT_MODEL as OPENAI_DEFAULT_MODEL
from .llm.openai_client import OpenAIClient

load_dotenv()


def build_llm_client(provider: str | None = None, model: str | None = None) -> LLMClient:
    provider = (provider or os.environ.get("LLM_PROVIDER") or "anthropic").lower()

    model = model or os.environ.get("LLM_MODEL")

    if provider == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set (required for provider='anthropic')")
        return AnthropicClient(api_key=api_key, model=model or ANTHROPIC_DEFAULT_MODEL)

    if provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set (required for provider='openai')")
        return OpenAIClient(api_key=api_key, model=model or OPENAI_DEFAULT_MODEL)

    raise ValueError(f"Unknown provider {provider!r}; expected 'anthropic' or 'openai'")
