"""Chunker adapter: fixed-size character windows that prefer natural boundaries.

Each section (PDF page, or a whole DOCX/TXT) is split on its own, so every
chunk has exactly one page number. Within a section, each chunk is at most
`chunk_size` characters and ends at the best boundary available, in order of
preference: a paragraph break, a line break, the end of a sentence, a space,
and only as a last resort the middle of a word. Consecutive chunks overlap by
about `chunk_overlap` characters (moved forward to a word start), so a
sentence cut by a boundary still appears whole in one of them.
"""

from collections.abc import Iterator, Sequence

from app.application.ports import Chunker
from app.domain.models import Chunk, Section

_BOUNDARIES: tuple[tuple[str, ...], ...] = (
    ("\n\n",),
    ("\n",),
    (". ", "! ", "? "),
    (" ",),
)


class RecursiveCharacterChunker(Chunker):
    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        if chunk_size < 1:
            raise ValueError("chunk_size must be at least 1.")
        if not 0 <= chunk_overlap < chunk_size:
            raise ValueError(
                "chunk_overlap must be at least 0 and less than chunk_size."
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, sections: Sequence[Section]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for section in sections:
            for start, end in self._spans(section.text):
                chunks.append(
                    Chunk(
                        index=len(chunks),
                        page_number=section.page_number,
                        text=section.text[start:end],
                        start_char=start,
                        end_char=end,
                    )
                )
        return chunks

    def _spans(self, text: str) -> Iterator[tuple[int, int]]:
        """Yield (start, end) positions of each chunk within `text`."""
        start = _skip_whitespace(text, 0)
        while start < len(text):
            limit = start + self.chunk_size
            end = (
                len(text)
                if limit >= len(text)
                else self._break_point(text, start, limit)
            )

            stop = end
            while stop > start and text[stop - 1].isspace():
                stop -= 1
            if stop > start:
                yield start, stop

            if end >= len(text):
                return
            next_start = _move_to_word_start(text, end - self.chunk_overlap, end)
            start = _skip_whitespace(text, next_start)

    def _break_point(self, text: str, start: int, limit: int) -> int:
        """Where to end a chunk that would otherwise run to `limit`.

        A break is only accepted past the halfway point and past the overlap,
        so chunks are never tiny and every chunk moves the window forward.
        """
        earliest = start + max(self.chunk_size // 2, self.chunk_overlap + 1)
        for separators in _BOUNDARIES:
            best = max(text.rfind(sep, earliest, limit) for sep in separators)
            if best != -1:
                return best + 1
        return limit


def _skip_whitespace(text: str, position: int) -> int:
    while position < len(text) and text[position].isspace():
        position += 1
    return position


def _move_to_word_start(text: str, position: int, end: int) -> int:
    """If `position` is inside a word, move to the start of the next word (before `end`)."""
    if position <= 0 or text[position - 1].isspace() or text[position].isspace():
        return position
    for i in range(position, end):
        if text[i].isspace():
            return i + 1
    return position
