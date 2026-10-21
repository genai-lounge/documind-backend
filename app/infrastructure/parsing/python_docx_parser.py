"""DOCX adapter for the DocumentParser port, using the python-docx library."""

import io

import docx
from docx.table import Table

from app.application.ports import DocumentParser
from app.domain.errors import DocumentParseError
from app.domain.models import DocumentMetadata, ParsedContent, Section


class PythonDocxParser(DocumentParser):
    format_name = "Word document"

    def matches_content(self, data: bytes) -> bool:
        return data.startswith(b"PK\x03\x04")

    def parse(self, data: bytes) -> ParsedContent:
        """Extract paragraphs and tables in the order they appear in the document.

        Word works out page breaks only when it lays the document out for display,
        so a DOCX file has no reliable page numbers. The text is one section.
        """
        try:
            document = docx.Document(io.BytesIO(data))
        except Exception as exc:
            raise DocumentParseError(
                "The DOCX file is damaged or not a valid Word document."
            ) from exc

        blocks: list[str] = []
        for item in document.iter_inner_content():
            if isinstance(item, Table):
                blocks.extend(_table_rows(item))
            elif item.text.strip():
                blocks.append(item.text)

        properties = document.core_properties
        return ParsedContent(
            sections=(Section(page_number=None, text="\n".join(blocks)),),
            metadata=DocumentMetadata(
                title=properties.title or None,
                author=properties.author or None,
            ),
        )


def _table_rows(table: Table) -> list[str]:
    """Turn each table row into one line, with cells separated by " | "."""
    rows = []
    for row in table.rows:
        cells: list[str] = []
        for cell in row.cells:
            text = cell.text.strip()
            if cells and cells[-1] == text:
                continue
            cells.append(text)
        if any(cells):
            rows.append(" | ".join(cells))
    return rows
