"""Unit tests for the IngestDocument use case, using fake parsers.

No real PDF library is involved: these tests check only the use case's own
rules (validation order, limits, status, warnings).
"""

import io
import logging

import pytest

from app.application.ingest_document import NO_TEXT_WARNING, IngestDocument
from app.application.parser_registry import ParserRegistry
from app.domain.errors import (
    DocumentParseError,
    EmbeddingUnavailable,
    EmptyFile,
    FileTooLarge,
    FileTypeMismatch,
    MissingFilename,
    UnsupportedFileType,
)
from app.domain.models import (
    DocumentMetadata,
    ExtractionStatus,
    ParsedContent,
    Section,
)
from tests.fakes import FakeChunker, FakeEmbeddingProvider, FakeParser


def make_use_case(
    parser: FakeParser | None = None,
    max_upload_bytes: int = 100,
    extensions: tuple[str, ...] = (".fake",),
    embedder: FakeEmbeddingProvider | None = None,
) -> IngestDocument:
    parser = parser or FakeParser()
    return IngestDocument(
        parsers=ParserRegistry({ext: parser for ext in extensions}),
        chunker=FakeChunker(),
        embedder=embedder or FakeEmbeddingProvider(),
        max_upload_bytes=max_upload_bytes,
        new_id=lambda: "fixed-id",
    )


def stream(data: bytes) -> io.BytesIO:
    return io.BytesIO(data)


def test_builds_document_from_parsed_content() -> None:
    content = ParsedContent(
        sections=(Section(1, "hello"),),
        page_count=1,
        metadata=DocumentMetadata(title="T"),
    )

    document = make_use_case(FakeParser(content)).execute("report.fake", stream(b"abc"))

    assert document.id == "fixed-id"
    assert document.filename == "report.fake"
    assert document.file_type == "fake"
    assert document.size_bytes == 3
    assert document.status is ExtractionStatus.EXTRACTED
    assert document.content == content


def test_strips_directories_from_filename() -> None:
    document = make_use_case().execute("../../etc/report.fake", stream(b"abc"))

    assert document.filename == "report.fake"


def test_extension_lookup_is_case_insensitive() -> None:
    assert make_use_case().execute("REPORT.FAKE", stream(b"abc")).file_type == "fake"


@pytest.mark.parametrize("filename", ["", "/"])
def test_missing_filename(filename: str) -> None:
    with pytest.raises(MissingFilename):
        make_use_case().execute(filename, stream(b"abc"))


def test_unsupported_extension_lists_supported_formats() -> None:
    use_case = make_use_case(extensions=(".pdf", ".docx", ".txt"))

    with pytest.raises(UnsupportedFileType) as error:
        use_case.execute("image.png", stream(b"abc"))

    assert error.value.message == (
        "'.png' files are not supported. Upload a PDF, DOCX, or TXT file."
    )


def test_name_without_extension_is_unsupported() -> None:
    with pytest.raises(UnsupportedFileType, match="'README'"):
        make_use_case().execute("README", stream(b"abc"))


def test_unsupported_extension_is_rejected_before_reading_the_upload() -> None:
    upload = stream(b"abc")

    with pytest.raises(UnsupportedFileType):
        make_use_case().execute("image.png", upload)

    assert upload.tell() == 0


def test_file_over_limit() -> None:
    with pytest.raises(FileTooLarge, match="10 bytes"):
        make_use_case(max_upload_bytes=10).execute("a.fake", stream(b"x" * 11))


def test_limit_message_uses_megabytes_when_whole() -> None:
    one_mb = 1024 * 1024
    with pytest.raises(FileTooLarge, match="1 MB limit"):
        make_use_case(max_upload_bytes=one_mb).execute(
            "a.fake", stream(b"x" * (one_mb + 1))
        )


def test_file_exactly_at_limit_is_accepted() -> None:
    document = make_use_case(max_upload_bytes=10).execute("a.fake", stream(b"x" * 10))

    assert document.size_bytes == 10


def test_never_reads_more_than_one_byte_past_the_limit() -> None:
    upload = stream(b"x" * 1000)

    with pytest.raises(FileTooLarge):
        make_use_case(max_upload_bytes=10).execute("a.fake", upload)

    assert upload.tell() == 11


def test_empty_file() -> None:
    with pytest.raises(EmptyFile):
        make_use_case().execute("a.fake", stream(b""))


def test_content_mismatch_is_rejected_without_parsing() -> None:
    parser = FakeParser(signature_matches=False)

    with pytest.raises(FileTypeMismatch) as error:
        make_use_case(parser).execute("a.fake", stream(b"abc"))

    assert (
        error.value.message == "The file has a .fake name but is not a fake document."
    )
    assert parser.parsed == []


def test_parse_errors_propagate() -> None:
    parser = FakeParser(error=DocumentParseError("broken"))

    with pytest.raises(DocumentParseError, match="broken"):
        make_use_case(parser).execute("a.fake", stream(b"abc"))


def test_no_text_gets_status_and_generic_warning() -> None:
    parser = FakeParser(ParsedContent(sections=(Section(None, "  \n "),)))

    document = make_use_case(parser).execute("a.fake", stream(b"abc"))

    assert document.status is ExtractionStatus.NO_TEXT
    assert document.content.warnings == (NO_TEXT_WARNING,)


def test_parser_warnings_replace_the_generic_warning() -> None:
    content = ParsedContent(sections=(Section(1, ""),), warnings=("Scanned PDF?",))

    document = make_use_case(FakeParser(content)).execute("a.fake", stream(b"abc"))

    assert document.status is ExtractionStatus.NO_TEXT
    assert document.content.warnings == ("Scanned PDF?",)


def test_logs_ingestion(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="app.application.ingest_document"):
        make_use_case().execute("a.fake", stream(b"abc"))

    record = next(r for r in caplog.records if r.message == "document.ingested")
    assert record.__dict__["document_id"] == "fixed-id"
    assert record.__dict__["size_bytes"] == 3


# --- Chunking and embedding ---


def test_chunks_are_embedded_in_order() -> None:
    content = ParsedContent(sections=(Section(1, "page one"), Section(2, "page two")))
    embedder = FakeEmbeddingProvider()

    document = make_use_case(FakeParser(content), embedder=embedder).execute(
        "a.fake", stream(b"abc")
    )

    assert [c.chunk.text for c in document.chunks] == ["page one", "page two"]
    assert [c.chunk.page_number for c in document.chunks] == [1, 2]
    assert [c.vector for c in document.chunks] == embedder.embed_documents(
        ["page one", "page two"]
    )
    assert document.embedding_model == embedder.model


def test_all_chunks_are_embedded_in_one_call() -> None:
    content = ParsedContent(sections=(Section(1, "one"), Section(2, "two")))
    embedder = FakeEmbeddingProvider()

    make_use_case(FakeParser(content), embedder=embedder).execute(
        "a.fake", stream(b"abc")
    )

    assert embedder.calls == [("document", ["one", "two"])]


def test_no_text_means_no_chunks_and_no_embedding_call() -> None:
    parser = FakeParser(ParsedContent(sections=(Section(1, "   "),)))
    embedder = FakeEmbeddingProvider()

    document = make_use_case(parser, embedder=embedder).execute(
        "a.fake", stream(b"abc")
    )

    assert document.chunks == ()
    assert document.embedding_model is None
    assert embedder.calls == []


def test_embedding_errors_propagate() -> None:
    embedder = FakeEmbeddingProvider(error=EmbeddingUnavailable("down"))

    with pytest.raises(EmbeddingUnavailable):
        make_use_case(embedder=embedder).execute("a.fake", stream(b"abc"))


def test_invalid_uploads_never_reach_the_embedding_service() -> None:
    embedder = FakeEmbeddingProvider()
    parser = FakeParser(error=DocumentParseError("broken"))

    with pytest.raises(DocumentParseError):
        make_use_case(parser, embedder=embedder).execute("a.fake", stream(b"abc"))

    assert embedder.calls == []
