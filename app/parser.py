from __future__ import annotations

from pathlib import Path

from app.ingest.pipeline import IngestionPipeline
from app.ingest.sources import LocalCorpusSource
from app.models import RawDocument


def load_documents(docs_dir: Path) -> list[RawDocument]:
    """Backward-compatible helper; prefer IngestionPipeline.parse_to_raw()."""
    return IngestionPipeline(source=LocalCorpusSource(docs_dir)).parse_to_raw()


def parse_markdown(path: Path) -> RawDocument:
    from app.ingest.parsers.markdown import MarkdownLayoutParser
    from app.ingest.types import DocumentArtifact

    artifact = DocumentArtifact(
        artifact_id=path.stem,
        uri=str(path.resolve()),
        mime_type="text/markdown",
        local_path=str(path),
        metadata={"suffix": path.suffix.lower()},
    )
    return MarkdownLayoutParser().parse(artifact).to_raw_document()
