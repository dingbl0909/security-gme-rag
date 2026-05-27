from __future__ import annotations

from app.database import RagDatabase
from app.evaluation import context_precision, context_recall, faithfulness, response_relevancy
from app.models import RetrievalHit
from app.workflow_types import RagasMetrics


class RagasEvaluationLoop:
    """Lightweight RAGAS-style evaluation loop.

    This class does not require the official ragas package. It records the same
    kind of signals used in a production RAGAS loop so the project can run
    locally while keeping a clear replacement point for official RAGAS.
    """

    def __init__(self, database: RagDatabase | None = None):
        self.database = database or RagDatabase()
        self.database.init_schema()

    def evaluate(self, query: str, answer: str, hits: list[RetrievalHit]) -> RagasMetrics:
        return RagasMetrics(
            context_precision=context_precision(query, hits),
            context_recall=context_recall(query, hits),
            response_relevancy=response_relevancy(query, answer),
            faithfulness=faithfulness(answer, hits),
        )

    def record(
        self,
        thread_id: str,
        query: str,
        answer: str,
        hits: list[RetrievalHit],
        metrics: RagasMetrics,
        needs_review: bool,
        review_reason: str | None,
    ) -> int:
        evidence = [
            {
                "chunk_id": hit.chunk.chunk_id,
                "title": hit.chunk.title,
                "source": hit.chunk.source,
                "modality": hit.chunk.modality,
                "image_ref": hit.chunk.image_ref,
                "dense_score": hit.dense_score,
                "sparse_score": hit.sparse_score,
                "hybrid_score": hit.hybrid_score,
            }
            for hit in hits
        ]
        return self.database.save_ragas_run(
            thread_id=thread_id,
            query=query,
            answer=answer,
            evidence=evidence,
            metrics=metrics.__dict__,
            needs_review=needs_review,
            review_reason=review_reason,
        )

    def recent_runs(self, limit: int = 20) -> list[dict]:
        return self.database.list_ragas_runs(limit=limit)

