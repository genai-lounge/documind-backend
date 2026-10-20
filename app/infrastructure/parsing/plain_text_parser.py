"""Plain-text adapter for the DocumentParser port, using only the standard library."""

from app.application.ports import DocumentParser
from app.domain.errors import DocumentParseError
from app.domain.models import ParsedContent, Section


class PlainTextParser(DocumentParser):
    format_name = "text file"

    def matches_content(self, data: bytes) -> bool:
        return True

    def parse(self, data: bytes) -> ParsedContent:
        """Decode a UTF-8 text file (a leading byte-order mark is allowed)."""
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise DocumentParseError(
                "The text file is not UTF-8 encoded. Save it as UTF-8 and upload it again."
            ) from exc

        if "\x00" in text:
            raise DocumentParseError(
                "The file looks like binary data, not a text file."
            )

        text = text.replace("\r\n", "\n")
        return ParsedContent(sections=(Section(page_number=None, text=text),))
