"""Question answering: the evidence sent to a language model, what it returns,
and the checked answer DocuMind serves.

A model's citations are claims, not facts: `GeneratedAnswer` holds what the
model said it cited, and only citations that match the evidence actually sent
become `Citation`s in the final `Answer`.
"""

from dataclasses import dataclass

from app.domain.models import SearchHit


@dataclass(frozen=True)
class EvidencePassage:
    """One retrieved chunk as shown to the model. Its position in the list sent is its index."""

    source_id: str
    title: str
    text: str


@dataclass(frozen=True)
class CitedSpan:
    """The model's claim that a segment is supported by passage `passage_index`."""

    passage_index: int
    quoted_text: str


@dataclass(frozen=True)
class AnswerSegment:
    text: str
    cited: tuple[CitedSpan, ...] = ()


@dataclass(frozen=True)
class GeneratedAnswer:
    """The model's raw answer, before its citations are checked."""

    segments: tuple[AnswerSegment, ...]
    model: str


@dataclass(frozen=True)
class Citation:
    """A checked citation: `number` is the [n] marker shown in the answer text."""

    number: int
    hit: SearchHit
    quote: str


@dataclass(frozen=True)
class Answer:
    """The answer served to the user.

    `insufficient_evidence` is True when no retrieved passage supports the
    answer; `text` then explains that the documents don't answer the question.
    `retrieved` lists every passage that was given to the model, cited or not.
    """

    text: str
    citations: tuple[Citation, ...]
    insufficient_evidence: bool
    model: str | None
    retrieved: tuple[SearchHit, ...]


@dataclass(frozen=True)
class ChatTurn:
    """An earlier question and answer in the same conversation (context, never evidence)."""

    question: str
    answer: str


# --- What a language model emits while streaming (unchecked) ---


@dataclass(frozen=True)
class TextChunk:
    text: str


@dataclass(frozen=True)
class CitationClaim:
    span: CitedSpan


@dataclass(frozen=True)
class BlockEnd:
    """The end of a run of text that the claims since the previous BlockEnd refer to."""


LLMEvent = TextChunk | CitationClaim | BlockEnd


# --- What DocuMind streams to clients (checked) ---


@dataclass(frozen=True)
class AnswerTextDelta:
    text: str


@dataclass(frozen=True)
class AnswerCitationAdded:
    citation: Citation


@dataclass(frozen=True)
class AnswerDone:
    insufficient_evidence: bool
    model: str | None


AnswerEvent = AnswerTextDelta | AnswerCitationAdded | AnswerDone
