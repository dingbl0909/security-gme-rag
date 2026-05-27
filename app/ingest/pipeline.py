from __future__ import annotations

from dataclasses import dataclass

from app.chunker import chunk_documents
from app.config import Settings, get_settings
from app.index import LocalIndex
from app.ingest.image_sidecars import build_image_sidecar_chunks
from app.ingest.parsers.factory import build_document_parser
from app.ingest.sources import DocumentSource, LocalCorpusSource
from app.ingest.types import DocumentArtifact, ParsedDocument
from app.models import KnowledgeChunk, RawDocument


@dataclass
class IngestionReport:
    artifact_count: int
    parsed_count: int
    chunk_count: int
    skipped: list[str]


class IngestionPipeline:
    """Collect artifacts -> parse with Dots.OCR/markdown -> chunk -> index."""

    def __init__(
        self,
        source: DocumentSource | None = None,
        parser=None,
        settings: Settings | None = None,
    ):
        self.settings = settings or get_settings()
        self.source = source or LocalCorpusSource(self.settings.docs_dir)
        self.parser = parser or build_document_parser(self.settings)

    def run(self, *, rebuild_index: bool = True) -> IngestionReport:
        artifacts = self.source.discover()
        parsed_docs, skipped = self._parse_all(artifacts)
        raw_documents = [doc.to_raw_document() for doc in parsed_docs]
        chunks = chunk_documents(raw_documents)
        chunks.extend(build_image_sidecar_chunks(self.settings.images_dir))
        if rebuild_index:
            LocalIndex().rebuild(chunks)
        return IngestionReport(
            artifact_count=len(artifacts),
            parsed_count=len(parsed_docs),
            chunk_count=len(chunks),
            skipped=skipped,
        )

    def parse_to_raw(self) -> list[RawDocument]:
        artifacts = self.source.discover()
        parsed_docs, _ = self._parse_all(artifacts)
        return [doc.to_raw_document() for doc in parsed_docs]

    def parse_and_chunk(self) -> list[KnowledgeChunk]:
        return chunk_documents(self.parse_to_raw())

    def ingest_artifact(self, artifact: DocumentArtifact, *, index: bool = False) -> tuple[ParsedDocument, list[KnowledgeChunk]]:
        parsed = self.parser.parse(artifact)
        chunks = chunk_documents([parsed.to_raw_document()])
        if index:
            indexer = LocalIndex()
            for chunk in chunks:
                vector = indexer.embedder.embed(chunk.content, chunk.image_ref)
                indexer.database.upsert_chunk(chunk, vector)
                indexer.vector_store.upsert(chunk, vector)
        return parsed, chunks

    def _parse_all(self, artifacts: list[DocumentArtifact]) -> tuple[list[ParsedDocument], list[str]]:
        parsed: list[ParsedDocument] = []
        skipped: list[str] = []
        for artifact in artifacts:
            try:
                parsed.append(self.parser.parse(artifact))
            except Exception as exc:  # pragma: no cover - surfaced in ingest report
                skipped.append(f"{artifact.uri}: {exc}")
        return parsed, skipped
