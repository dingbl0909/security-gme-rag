from __future__ import annotations

from app.config import get_settings
from app.database import RagDatabase
from app.embedding import build_embedder
from app.models import KnowledgeChunk
from app.vector_store import build_vector_store


class LocalIndex:
    def __init__(self, database: RagDatabase | None = None, embedder=None):
        self.settings = get_settings()
        self.database = database or RagDatabase()
        self.embedder = embedder or build_embedder(self.settings)
        self.vector_store = build_vector_store(self.settings, self.database)
        self.database.init_schema()

    def rebuild(self, chunks: list[KnowledgeChunk]) -> int:
        self.database.clear_chunks()
        self.vector_store.reset()
        for chunk in chunks:
            vector = self.embedder.embed(chunk.content, chunk.image_ref)
            self.database.upsert_chunk(chunk, vector)
            self.vector_store.upsert(chunk, vector)
        return len(chunks)

