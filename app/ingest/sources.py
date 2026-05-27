from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Protocol

from app.ingest.types import DocumentArtifact


SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt", ".pdf", ".ppt", ".pptx", ".doc", ".docx", ".html", ".htm"}


class DocumentSource(Protocol):
    def discover(self) -> list[DocumentArtifact]:
        ...


class LocalCorpusSource:
    """Scan a local directory for ingestible security knowledge files."""

    def __init__(self, root: Path, *, include_suffixes: set[str] | None = None):
        self.root = root
        self.include_suffixes = include_suffixes or SUPPORTED_SUFFIXES

    def discover(self) -> list[DocumentArtifact]:
        if not self.root.exists():
            return []
        artifacts: list[DocumentArtifact] = []
        for path in sorted(self.root.rglob("*")):
            if not path.is_file():
                continue
            suffix = path.suffix.lower()
            if suffix not in self.include_suffixes:
                continue
            mime_type, _ = mimetypes.guess_type(path.name)
            artifacts.append(
                DocumentArtifact(
                    artifact_id=path.stem,
                    uri=str(path.resolve()),
                    mime_type=mime_type or "application/octet-stream",
                    local_path=str(path),
                    metadata={"suffix": suffix, "relative_path": str(path.relative_to(self.root))},
                )
            )
        return artifacts
