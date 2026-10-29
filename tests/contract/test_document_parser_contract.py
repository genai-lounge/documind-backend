"""Contract tests: rules EVERY DocumentParser adapter must follow.

The use case relies on these rules, so any parser that passes them can be
swapped in without surprises (the Liskov Substitution Principle). The parsers
are taken from the real composition root, so registering a new parser without
adding samples here makes `test_every_registered_parser_has_samples` fail.
"""

from dataclasses import dataclass

import pytest

from app.application.ports import DocumentParser
from app.container import Container
from app.core.config import Settings
from app.domain.errors import DocumentParseError
from app.domain.models import ParsedContent
from app.infrastructure.parsing.plain_text_parser import PlainTextParser
from app.infrastructure.parsing.pymupdf_parser import PyMuPdfParser
from app.infrastructure.parsing.python_docx_parser import PythonDocxParser
from tests.factories import make_docx, make_pdf
from tests.fakes import FakeEmbeddingProvider, FakeLLMProvider


@dataclass(frozen=True)
class Samples:
    valid: bytes
    damaged: bytes


SAMPLES: dict[type[DocumentParser], Samples] = {
    PyMuPdfParser: Samples(valid=make_pdf(["hello"]), damaged=b"%PDF-1.7 garbage"),
    PythonDocxParser: Samples(
        valid=make_docx(["hello"]), damaged=b"PK\x03\x04 garbage"
    ),
    PlainTextParser: Samples(valid=b"hello", damaged=b"\xff\xfe\xfa invalid utf-8"),
}

REGISTRY = Container(
    Settings(_env_file=None),
    embedding_provider=FakeEmbeddingProvider(),
    llm_provider=FakeLLMProvider(),
).parser_registry
REGISTERED = [REGISTRY.get(ext) for ext in REGISTRY.supported_extensions]


def test_every_registered_parser_has_samples() -> None:
    for parser in REGISTERED:
        assert type(parser) in SAMPLES, (
            f"Add contract samples for {type(parser).__name__}"
        )


@pytest.fixture(params=list(SAMPLES), ids=lambda cls: cls.__name__)
def parser_and_samples(
    request: pytest.FixtureRequest,
) -> tuple[DocumentParser, Samples]:
    parser_class: type[DocumentParser] = request.param
    return parser_class(), SAMPLES[parser_class]


def test_has_a_format_name(parser_and_samples: tuple[DocumentParser, Samples]) -> None:
    parser, _ = parser_and_samples
    assert isinstance(parser.format_name, str) and parser.format_name


def test_accepts_its_own_format(
    parser_and_samples: tuple[DocumentParser, Samples],
) -> None:
    parser, samples = parser_and_samples
    assert parser.matches_content(samples.valid)


def test_parses_valid_input_into_sections(
    parser_and_samples: tuple[DocumentParser, Samples],
) -> None:
    parser, samples = parser_and_samples

    result = parser.parse(samples.valid)

    assert isinstance(result, ParsedContent)
    assert len(result.sections) >= 1
    assert result.has_text
    assert all(s.page_number is None or s.page_number >= 1 for s in result.sections)


def test_damaged_input_raises_only_document_parse_error(
    parser_and_samples: tuple[DocumentParser, Samples],
) -> None:
    """Library-specific exceptions must never escape an adapter."""
    parser, samples = parser_and_samples

    with pytest.raises(DocumentParseError):
        parser.parse(samples.damaged)
