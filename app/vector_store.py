from __future__ import annotations

from app.config import Settings
from app.database import RagDatabase
from app.models import KnowledgeChunk


class SQLiteVectorStore:
    """No-op vector store because vectors are already persisted in SQLite."""

    def __init__(self, database: RagDatabase):
        self.database = database

    def reset(self) -> None:
        return None

    def upsert(self, chunk: KnowledgeChunk, vector: list[float]) -> None:
        return None

    def search(self, query_vector: list[float], top_k: int) -> dict[str, float]:
        return {}


class MilvusVectorStore:
    def __init__(self, settings: Settings):
        self.settings = settings
        try:
            from pymilvus import MilvusClient
        except ImportError as exc:  # pragma: no cover - optional production dependency
            raise RuntimeError("pymilvus is required when GME_RAG_VECTOR_STORE=milvus") from exc
        self.client = MilvusClient(uri=settings.milvus_uri, token=settings.milvus_token)
        self.collection = settings.milvus_collection
        self.dimensions = settings.gme_dimensions if settings.embedding_provider == "gme" else 64
        self._ensure_collection()

    def reset(self) -> None:
        if self.client.has_collection(self.collection):
            self.client.drop_collection(self.collection)
        self._ensure_collection()

    def upsert(self, chunk: KnowledgeChunk, vector: list[float]) -> None:
        self.client.upsert(
            collection_name=self.collection,
            data=[
                {
                    "id": chunk.chunk_id,
                    "vector": vector,
                    "doc_id": chunk.doc_id,
                    "title": chunk.title,
                    "modality": chunk.modality,
                    "image_ref": chunk.image_ref or "",
                }
            ],
        )

    def search(self, query_vector: list[float], top_k: int) -> dict[str, float]:
        results = self.client.search(
            collection_name=self.collection,
            data=[query_vector],
            limit=top_k,
            output_fields=["id"],
        )
        scores: dict[str, float] = {}
        for hit in results[0]:
            chunk_id = hit.get("id") or hit.get("entity", {}).get("id")
            score = hit.get("distance", hit.get("score", 0.0))
            if chunk_id:
                scores[str(chunk_id)] = float(score)
        return scores

    def _ensure_collection(self) -> None:
        if self.client.has_collection(self.collection):
            return
        self.client.create_collection(
            collection_name=self.collection,
            dimension=self.dimensions,
            metric_type="COSINE",
            auto_id=False,
        )


def build_vector_store(settings: Settings, database: RagDatabase):
    if settings.vector_store == "sqlite":
        return SQLiteVectorStore(database)
    if settings.vector_store == "milvus":
        return MilvusVectorStore(settings)
    raise ValueError(f"Unsupported vector store: {settings.vector_store}")

