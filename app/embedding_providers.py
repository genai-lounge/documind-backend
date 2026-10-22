"""Embedding provider registry: EMBEDDING_PROVIDER name -> builder function.

Part of the composition root. Each builder reads only its own vendor's
settings and turns configuration mistakes into ConfigurationError, so the app
refuses to start with a clear message.

Adding a vendor (Cohere, Gemini, Mistral, ...) means writing one adapter in
app/infrastructure/embeddings/ and adding one builder and one registry entry
here. Nothing else changes (Open/Closed Principle).
"""

from collections.abc import Callable

from app.application.ports import EmbeddingProvider
from app.core.config import ConfigurationError, Settings
from app.infrastructure.embeddings import (
    fastembed_provider,
    openai_embedding_provider,
    voyage_embedding_provider,
)

EmbeddingProviderBuilder = Callable[[Settings], EmbeddingProvider]


def _required_key(value: object, variable: str, where: str) -> str:
    secret = getattr(value, "get_secret_value", lambda: "")()
    if not isinstance(secret, str) or not secret.strip():
        raise ConfigurationError(
            f"{variable} is not set. Create an API key at {where} "
            f"and add {variable}=... to backend/.env (see the README)."
        )
    return secret.strip()


def build_openai(settings: Settings) -> EmbeddingProvider:
    key = _required_key(
        settings.openai_api_key,
        "OPENAI_API_KEY",
        "https://platform.openai.com/api-keys",
    )
    model = settings.openai_embedding_model
    dimensions = settings.openai_embedding_dimensions
    native = openai_embedding_provider.NATIVE_DIMENSIONS.get(model)
    if dimensions is not None and native is not None:
        if model not in openai_embedding_provider.SHORTENABLE_MODELS:
            raise ConfigurationError(
                f"OPENAI_EMBEDDING_DIMENSIONS can't be set for {model}; leave it empty."
            )
        if dimensions > native:
            raise ConfigurationError(
                f"OPENAI_EMBEDDING_DIMENSIONS must be at most {native} for {model}."
            )
    if dimensions is None and native is None:
        raise ConfigurationError(
            f"Set OPENAI_EMBEDDING_DIMENSIONS for the unrecognised model {model}."
        )
    return openai_embedding_provider.OpenAIEmbeddingProvider(
        api_key=key,
        model_name=model,
        dimensions=dimensions,
        batch_size=settings.embedding_batch_size,
        timeout_seconds=settings.embedding_timeout_seconds,
        max_retries=settings.embedding_max_retries,
    )


def build_voyage(settings: Settings) -> EmbeddingProvider:
    key = _required_key(
        settings.voyage_api_key, "VOYAGE_API_KEY", "https://dash.voyageai.com"
    )
    supported = voyage_embedding_provider.SUPPORTED_DIMENSIONS
    if settings.voyage_dimensions not in supported:
        raise ConfigurationError(f"VOYAGE_DIMENSIONS must be one of {supported}.")
    return voyage_embedding_provider.VoyageEmbeddingProvider(
        api_key=key,
        model_name=settings.voyage_model,
        dimensions=settings.voyage_dimensions,
        batch_size=settings.embedding_batch_size,
        timeout_seconds=settings.embedding_timeout_seconds,
        max_retries=settings.embedding_max_retries,
    )


def build_fastembed(settings: Settings) -> EmbeddingProvider:
    try:
        return fastembed_provider.FastEmbedProvider(
            model_name=settings.fastembed_model,
            cache_dir=str(settings.fastembed_cache_dir),
            batch_size=settings.embedding_batch_size,
        )
    except (ImportError, ValueError) as exc:
        raise ConfigurationError(str(exc)) from exc


EMBEDDING_PROVIDERS: dict[str, EmbeddingProviderBuilder] = {
    "openai": build_openai,
    "voyage": build_voyage,
    "fastembed": build_fastembed,
}


def build_embedding_provider(settings: Settings) -> EmbeddingProvider:
    name = settings.embedding_provider.strip().lower()
    builder = EMBEDDING_PROVIDERS.get(name)
    if builder is None:
        raise ConfigurationError(
            f"Unknown EMBEDDING_PROVIDER '{settings.embedding_provider}'. "
            f"Choose one of: {', '.join(EMBEDDING_PROVIDERS)}."
        )
    return builder(settings)
