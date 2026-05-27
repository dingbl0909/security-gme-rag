from __future__ import annotations

from typing import Protocol

from app.ingest.types import DocumentArtifact, ParsedDocument


class DocumentParser(Protocol):
    """Parse a discovered artifact into a layout-aware ParsedDocument."""

    def parse(self, artifact: DocumentArtifact) -> ParsedDocument:
        ...
