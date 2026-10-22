"""Contract tests: rules EVERY Chunker must follow (see the Chunker port)."""

from collections.abc import Callable

import pytest

from app.application.ports import Chunker
from app.domain.models import Section
from app.infrastructure.chunking.recursive_character_chunker import (
    RecursiveCharacterChunker,
)
from tests.fakes import FakeChunker

CHUNKERS: dict[str, Callable[[], Chunker]] = {
    "RecursiveCharacterChunker": lambda: RecursiveCharacterChunker(200, 40),
    "FakeChunker": FakeChunker,
}

SECTIONS = [
    Section(1, "Revenue grew 12% in Q3. " * 30),
    Section(2, "   "),
    Section(3, "Outlook.\n\nCosts fell. " * 20),
    Section(None, "A short closing note."),
]


@pytest.fixture(params=list(CHUNKERS))
def chunker(request: pytest.FixtureRequest) -> Chunker:
    return CHUNKERS[request.param]()


def test_chunks_are_numbered_from_zero(chunker: Chunker) -> None:
    chunks = chunker.chunk(SECTIONS)

    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_each_chunk_is_an_exact_slice_of_one_section(chunker: Chunker) -> None:
    by_page = {s.page_number: s for s in SECTIONS}

    for chunk in chunker.chunk(SECTIONS):
        section = by_page[chunk.page_number]
        assert chunk.text == section.text[chunk.start_char : chunk.end_char]


def test_no_empty_chunks_and_none_for_empty_sections(chunker: Chunker) -> None:
    chunks = chunker.chunk(SECTIONS)

    assert all(c.text.strip() for c in chunks)
    assert 2 not in {c.page_number for c in chunks}


def test_no_sections_gives_no_chunks(chunker: Chunker) -> None:
    assert chunker.chunk([]) == []
