"""Fake adapters for unit tests.

A fake implements a port with simple, predictable behaviour and no real
library behind it. Use-case tests run against fakes, so they are fast and
test only the use case's own logic.
"""

from app.application.ports import DocumentParser
from app.domain.models import ParsedContent, Section


class FakeParser(DocumentParser):
    format_name = "fake document"

    def __init__(
        self,
        content: ParsedContent | None = None,
        signature_matches: bool = True,
        error: Exception | None = None,
    ) -> None:
        self.content = content or ParsedContent(sections=(Section(1, "some text"),))
        self.signature_matches = signature_matches
        self.error = error
        self.parsed: list[bytes] = []

    def matches_content(self, data: bytes) -> bool:
        return self.signature_matches

    def parse(self, data: bytes) -> ParsedContent:
        self.parsed.append(data)
        if self.error:
            raise self.error
        return self.content
