"""Chooses the right DocumentParser for a file extension.

Which parser handles which extension is decided by the composition root, not
here. Supporting a new format means registering one more parser there; no code
in this module or in the use case changes (Open/Closed Principle).
"""

from collections.abc import Mapping

from app.application.ports import DocumentParser


class ParserRegistry:
    def __init__(self, parsers_by_extension: Mapping[str, DocumentParser]) -> None:
        for extension in parsers_by_extension:
            if not extension.startswith(".") or extension != extension.lower():
                raise ValueError(
                    f"Extensions must be lowercase and start with '.': {extension!r}"
                )
        self._parsers = dict(parsers_by_extension)

    @property
    def supported_extensions(self) -> tuple[str, ...]:
        """Registered extensions, in registration order (e.g. ".pdf", ".docx")."""
        return tuple(self._parsers)

    def get(self, extension: str) -> DocumentParser | None:
        """Return the parser for an extension (case-insensitive), or None."""
        return self._parsers.get(extension.lower())
