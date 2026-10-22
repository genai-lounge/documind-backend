"""Choosing and configuring the embedding vendor from Settings."""

import sys
from typing import Any

import pytest
from pydantic import SecretStr

from app.core.config import ConfigurationError, Settings
from app.embedding_providers import EMBEDDING_PROVIDERS, build_embedding_provider
from app.infrastructure.embeddings.fastembed_provider import NOT_INSTALLED
from app.infrastructure.embeddings.openai_embedding_provider import (
    OpenAIEmbeddingProvider,
)
from app.infrastructure.embeddings.voyage_embedding_provider import (
    VoyageEmbeddingProvider,
)


def settings(**values: Any) -> Settings:
    return Settings(_env_file=None, **values)


def test_registry_lists_every_supported_vendor() -> None:
    assert set(EMBEDDING_PROVIDERS) == {"openai", "voyage", "fastembed"}


def test_openai_is_the_default() -> None:
    provider = build_embedding_provider(settings(openai_api_key=SecretStr("sk-test")))

    assert isinstance(provider, OpenAIEmbeddingProvider)
    assert provider.model.name == "text-embedding-3-small"
    assert provider.model.dimensions == 1536


@pytest.mark.parametrize("name", ["voyage", "VOYAGE", " Voyage "])
def test_vendor_names_are_case_insensitive(name: str) -> None:
    provider = build_embedding_provider(
        settings(embedding_provider=name, voyage_api_key=SecretStr("pa-test"))
    )

    assert isinstance(provider, VoyageEmbeddingProvider)
    assert provider.model.dimensions == 1024


def test_unknown_vendor_lists_the_choices() -> None:
    with pytest.raises(ConfigurationError, match="openai, voyage, fastembed"):
        build_embedding_provider(settings(embedding_provider="cohere"))


def test_each_vendor_needs_only_its_own_key() -> None:
    with pytest.raises(ConfigurationError, match="VOYAGE_API_KEY"):
        build_embedding_provider(
            settings(embedding_provider="voyage", openai_api_key=SecretStr("sk-test"))
        )


def test_openai_dimensions_can_shorten_vectors() -> None:
    provider = build_embedding_provider(
        settings(openai_api_key=SecretStr("sk-test"), openai_embedding_dimensions=512)
    )

    assert provider.model.dimensions == 512


@pytest.mark.parametrize(
    ("model", "dimensions", "message"),
    [
        ("text-embedding-3-small", 2000, "at most 1536"),
        ("text-embedding-ada-002", 512, "can't be set"),
        ("my-custom-model", None, "Set OPENAI_EMBEDDING_DIMENSIONS"),
    ],
)
def test_invalid_openai_configuration(
    model: str, dimensions: int | None, message: str
) -> None:
    with pytest.raises(ConfigurationError, match=message):
        build_embedding_provider(
            settings(
                openai_api_key=SecretStr("sk-test"),
                openai_embedding_model=model,
                openai_embedding_dimensions=dimensions,
            )
        )


def test_voyage_dimensions_must_be_supported() -> None:
    with pytest.raises(ConfigurationError, match="VOYAGE_DIMENSIONS"):
        build_embedding_provider(
            settings(
                embedding_provider="voyage",
                voyage_api_key=SecretStr("pa-test"),
                voyage_dimensions=300,
            )
        )


def test_fastembed_without_the_extra_explains_how_to_install_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "fastembed", None)

    with pytest.raises(ConfigurationError) as error:
        build_embedding_provider(settings(embedding_provider="fastembed"))

    assert str(error.value) == NOT_INSTALLED


def test_fastembed_rejects_unknown_models() -> None:
    pytest.importorskip("fastembed")

    with pytest.raises(ConfigurationError, match="does not support"):
        build_embedding_provider(
            settings(embedding_provider="fastembed", fastembed_model="not/a-model")
        )
