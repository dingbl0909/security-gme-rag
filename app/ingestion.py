from __future__ import annotations

from app.ingest.pipeline import IngestionPipeline, IngestionReport


def ingest_demo_corpus() -> int:
    report = IngestionPipeline().run()
    return report.chunk_count


def ingest_demo_corpus_with_report() -> IngestionReport:
    return IngestionPipeline().run()
