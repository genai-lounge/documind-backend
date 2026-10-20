"""PDF adapter for the DocumentParser port, using the PyMuPDF library."""

import pymupdf

from app.application.ports import DocumentParser
from app.domain.errors import DocumentParseError
from app.domain.models import DocumentMetadata, ParsedContent, Section

NO_TEXT_IN_PDF_WARNING = (
    "No text was found in this PDF. It may be a scanned image; OCR is not supported."
)


class PyMuPdfParser(DocumentParser):
    format_name = "PDF"

    def matches_content(self, data: bytes) -> bool:
        return b"%PDF-" in data[:1024]

    def parse(self, data: bytes) -> ParsedContent:
        """Extract text page by page, keeping each page's number."""
        try:
            pdf = pymupdf.open(stream=data, filetype="pdf")
        except Exception as exc:
            raise DocumentParseError(
                "The PDF file is damaged or not a valid PDF."
            ) from exc

        with pdf:
            if pdf.needs_pass:
                raise DocumentParseError(
                    "The PDF is password-protected. Remove the password and upload it again."
                )

            try:
                sections = tuple(
                    Section(
                        page_number=number, text=pdf.load_page(number - 1).get_text()
                    )
                    for number in range(1, pdf.page_count + 1)
                )
            except Exception as exc:
                raise DocumentParseError(
                    "Text could not be read from the PDF's pages."
                ) from exc

            metadata = pdf.metadata or {}
            return ParsedContent(
                sections=sections,
                page_count=pdf.page_count,
                metadata=DocumentMetadata(
                    title=metadata.get("title") or None,
                    author=metadata.get("author") or None,
                ),
                warnings=_page_warnings(sections),
            )


def _page_warnings(sections: tuple[Section, ...]) -> tuple[str, ...]:
    if not any(section.has_text for section in sections):
        return (NO_TEXT_IN_PDF_WARNING,)
    return tuple(
        f"Page {section.page_number} has no extractable text "
        "(it may be a scanned image or blank)."
        for section in sections
        if not section.has_text
    )
