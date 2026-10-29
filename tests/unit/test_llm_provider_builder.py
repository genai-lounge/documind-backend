from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr

from app.core.config import ConfigurationError, Settings
from app.infrastructure.llm.claude_llm_provider import ClaudeLLMProvider
from app.llm_providers import build_llm_provider


def settings(tmp_path: Path, **values: Any) -> Settings:
    return Settings(_env_file=None, data_dir=tmp_path, **values)


def test_haiku_is_the_default_model(tmp_path: Path) -> None:
    provider = build_llm_provider(
        settings(tmp_path, anthropic_api_key=SecretStr("sk-ant-x"))
    )

    assert isinstance(provider, ClaudeLLMProvider)
    assert provider.model_name == "claude-haiku-5-5"


def test_model_is_configurable(tmp_path: Path) -> None:
    provider = build_llm_provider(
        settings(
            tmp_path,
            anthropic_api_key=SecretStr("sk-ant-x"),
            anthropic_model="claude-sonnet-5-5",
        )
    )

    assert provider.model_name == "claude-sonnet-5-5"


@pytest.mark.parametrize("key", [None, "", "   "])
def test_needs_an_api_key(tmp_path: Path, key: str | None) -> None:
    with pytest.raises(ConfigurationError, match="ANTHROPIC_API_KEY"):
        build_llm_provider(
            settings(
                tmp_path, anthropic_api_key=SecretStr(key) if key is not None else None
            )
        )


def test_effort_must_be_a_known_level(monkeypatch: pytest.MonkeyPatch) -> None:
    from pydantic import ValidationError

    monkeypatch.setenv("ANTHROPIC_EFFORT", "extreme")

    with pytest.raises(ValidationError):
        Settings(_env_file=None)
