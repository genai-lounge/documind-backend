import itertools

import pytest

from app.domain.models import Section
from app.infrastructure.chunking.recursive_character_chunker import (
    RecursiveCharacterChunker,
)


def chunk_texts(text: str, size: int = 100, overlap: int = 20) -> list[str]:
    chunker = RecursiveCharacterChunker(chunk_size=size, chunk_overlap=overlap)
    return [c.text for c in chunker.chunk([Section(1, text)])]


def test_short_text_is_one_trimmed_chunk() -> None:
    chunker = RecursiveCharacterChunker(chunk_size=100, chunk_overlap=20)

    [chunk] = chunker.chunk([Section(3, "  Hello world.  \n")])

    assert chunk.text == "Hello world."
    assert (chunk.start_char, chunk.end_char) == (2, 14)
    assert chunk.page_number == 3
    assert chunk.index == 0


@pytest.mark.parametrize("text", ["", "   ", "\n\n\t"])
def test_text_without_content_gives_no_chunks(text: str) -> None:
    assert chunk_texts(text) == []


def test_no_chunk_is_longer_than_chunk_size() -> None:
    text = " ".join(f"word{i}" for i in range(500))

    assert all(len(t) <= 100 for t in chunk_texts(text))


def test_prefers_paragraph_breaks() -> None:
    first = "First paragraph sentence one. Sentence two is here too."
    second = "Second paragraph " + "continues " * 10
    chunks = chunk_texts(f"{first}\n\n{second}", size=100, overlap=10)

    assert chunks[0] == first


def test_prefers_sentence_ends_over_spaces() -> None:
    text = "Alpha beta gamma delta epsilon zeta eta theta. " + "Iota kappa " * 10

    assert chunk_texts(text, size=80, overlap=10)[0].endswith("theta.")


def test_breaks_between_words_when_there_is_no_sentence_end() -> None:
    text = " ".join(["lorem"] * 60)

    for chunk in chunk_texts(text, size=50, overlap=10):
        assert chunk.split() == ["lorem"] * len(chunk.split())


def test_hard_cuts_a_single_huge_word() -> None:
    chunks = chunk_texts("x" * 250, size=100, overlap=20)

    assert [len(c) for c in chunks] == [100, 100, 90]


def test_consecutive_chunks_overlap_and_start_at_a_word() -> None:
    text = " ".join(f"w{i:03d}" for i in range(200))
    chunker = RecursiveCharacterChunker(chunk_size=100, chunk_overlap=30)

    chunks = chunker.chunk([Section(1, text)])

    for previous, current in itertools.pairwise(chunks):
        assert current.start_char < previous.end_char
        assert previous.end_char - current.start_char <= 30
        assert text[current.start_char - 1] == " "


def test_covers_all_the_text() -> None:
    text = " ".join(f"token{i}" for i in range(300))
    chunker = RecursiveCharacterChunker(chunk_size=120, chunk_overlap=25)

    covered: set[int] = set()
    for chunk in chunker.chunk([Section(None, text)]):
        covered.update(range(chunk.start_char, chunk.end_char))

    assert all(i in covered for i, ch in enumerate(text) if not ch.isspace())


def test_numbers_chunks_across_sections_and_keeps_page_numbers() -> None:
    chunker = RecursiveCharacterChunker(chunk_size=100, chunk_overlap=10)
    sections = [Section(1, "page one " * 20), Section(2, ""), Section(3, "page three")]

    chunks = chunker.chunk(sections)

    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert {c.page_number for c in chunks} == {1, 3}
    assert chunks[-1].page_number == 3


def test_is_deterministic() -> None:
    text = "Sentence number one. " * 100

    assert chunk_texts(text) == chunk_texts(text)


def test_always_makes_progress_with_maximum_overlap() -> None:
    chunks = chunk_texts("abcdefghij " * 50, size=100, overlap=99)

    assert 0 < len(chunks) < 1000


@pytest.mark.parametrize(("size", "overlap"), [(0, 0), (100, 100), (100, -1)])
def test_rejects_invalid_settings(size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        RecursiveCharacterChunker(chunk_size=size, chunk_overlap=overlap)
