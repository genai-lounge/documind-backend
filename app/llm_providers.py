"""Builds the language model client from Settings.

Part of the composition root. Configuration problems become ConfigurationError,
so the app refuses to start with a clear message. Another vendor would be one
adapter in app/infrastructure/llm/ plus one builder here.
"""

from app.application.ports import LLMProvider
from app.core.config import ConfigurationError, Settings
from app.infrastructure.llm.claude_llm_provider import ClaudeLLMProvider


def build_llm_provider(settings: Settings) -> LLMProvider:
    key = settings.anthropic_api_key
    if key is None or not key.get_secret_value().strip():
        raise ConfigurationError(
            "ANTHROPIC_API_KEY is not set. Create an API key at "
            "https://console.anthropic.com (API Keys) and add ANTHROPIC_API_KEY=... "
            "to backend/.env (not .env.example)."
        )
    return ClaudeLLMProvider(
        api_key=key.get_secret_value().strip(),
        model_name=settings.anthropic_model,
        effort=settings.anthropic_effort,
        max_tokens=settings.anthropic_max_tokens,
        timeout_seconds=settings.anthropic_timeout_seconds,
        max_retries=settings.anthropic_max_retries,
    )
