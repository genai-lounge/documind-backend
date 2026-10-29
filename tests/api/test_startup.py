"""The app refuses to start without the configuration it needs (fail fast)."""

import pytest
from pydantic import SecretStr

from app.core.config import ConfigurationError, Settings
from app.main import create_app


@pytest.mark.parametrize("key", [None, "", "   "])
def test_default_provider_openai_needs_its_key(key: str | None) -> None:
    settings = Settings(
        _env_file=None, openai_api_key=SecretStr(key) if key is not None else None
    )

    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
        create_app(settings)


def test_starts_with_a_key_without_calling_the_vendor() -> None:
    app = create_app(
        Settings(
            _env_file=None,
            openai_api_key=SecretStr("sk-test"),
            anthropic_api_key=SecretStr("sk-ant-test"),
        )
    )

    assert app.state.compare_texts is not None


@pytest.mark.parametrize("key", [None, "", "   "])
def test_refuses_to_start_without_an_anthropic_key(key: str | None) -> None:
    settings = Settings(
        _env_file=None,
        openai_api_key=SecretStr("sk-test"),
        anthropic_api_key=SecretStr(key) if key is not None else None,
    )

    with pytest.raises(ConfigurationError, match="ANTHROPIC_API_KEY"):
        create_app(settings)
