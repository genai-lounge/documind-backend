import pytest

from app.application.parser_registry import ParserRegistry
from tests.fakes import FakeParser


def test_lookup_is_case_insensitive() -> None:
    parser = FakeParser()
    registry = ParserRegistry({".pdf": parser})

    assert registry.get(".PDF") is parser


def test_unknown_extension_returns_none() -> None:
    assert ParserRegistry({".pdf": FakeParser()}).get(".png") is None


def test_keeps_registration_order() -> None:
    registry = ParserRegistry({".pdf": FakeParser(), ".txt": FakeParser()})

    assert registry.supported_extensions == (".pdf", ".txt")


@pytest.mark.parametrize("bad_extension", ["pdf", ".PDF", ""])
def test_rejects_badly_formatted_extensions(bad_extension: str) -> None:
    with pytest.raises(ValueError):
        ParserRegistry({bad_extension: FakeParser()})
