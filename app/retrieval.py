from __future__ import annotations

import math
import re

from app.config import get_settings
from app.database import RagDatabase
from app.embedding import build_embedder, cosine_similarity
from app.models import KnowledgeChunk, RetrievalHit
from app.vector_store import build_vector_store


TOKEN_RE = re.compile(r"[\u4e00-\u9fff]{2,}|[a-zA-Z0-9_./:-]+")


class HybridRetriever:
    def __init__(self, database: RagDatabase | None = None, embedder=None):
        self.settings = get_settings()
        self.database = database or RagDatabase(self.settings.db_path)
        self.embedder = embedder or build_embedder(self.settings)
        self.vector_store = build_vector_store(self.settings, self.database)

    def search(self, query: str, image_ref: str | None = None, top_k: int | None = None) -> list[RetrievalHit]:
        indexed = self.database.load_chunks()
        if not indexed:
            return []
        query_vector = self.embedder.embed(query, image_ref)
        milvus_dense_scores = self.vector_store.search(query_vector, top_k or self.settings.top_k)
        query_tokens = tokenize(query)
        document_frequencies = _document_frequencies([chunk for chunk, _ in indexed])
        hits: list[RetrievalHit] = []
        for chunk, vector in indexed:
            dense_score = milvus_dense_scores.get(chunk.chunk_id, max(cosine_similarity(query_vector, vector), 0.0))
            sparse_score = bm25_score(query_tokens, chunk, document_frequencies, len(indexed))
            hybrid_score = self.settings.dense_weight * dense_score + self.settings.sparse_weight * sparse_score
            if image_ref and chunk.image_ref == image_ref:
                hybrid_score += 0.5
            if hybrid_score > 0:
                hits.append(
                    RetrievalHit(
                        chunk=chunk,
                        dense_score=round(dense_score, 4),
                        sparse_score=round(sparse_score, 4),
                        hybrid_score=round(hybrid_score, 4),
                    )
                )
        hits.sort(key=lambda hit: hit.hybrid_score, reverse=True)
        return hits[: top_k or self.settings.top_k]


def tokenize(text: str) -> list[str]:
    lowered = text.lower()
    tokens = [match.group(0) for match in TOKEN_RE.finditer(lowered)]
    tokens.extend([char for char in lowered if "\u4e00" <= char <= "\u9fff"])
    return tokens


def bm25_score(query_tokens: list[str], chunk: KnowledgeChunk, document_frequencies: dict[str, int], doc_count: int) -> float:
    chunk_tokens = tokenize(chunk.content + " " + chunk.title)
    if not query_tokens or not chunk_tokens:
        return 0.0
    counts: dict[str, int] = {}
    for token in chunk_tokens:
        counts[token] = counts.get(token, 0) + 1
    avgdl = 120.0
    k1 = 1.5
    b = 0.75
    score = 0.0
    for token in query_tokens:
        tf = counts.get(token, 0)
        if tf == 0:
            continue
        df = document_frequencies.get(token, 1)
        idf = math.log(1 + (doc_count - df + 0.5) / (df + 0.5))
        denominator = tf + k1 * (1 - b + b * len(chunk_tokens) / avgdl)
        score += idf * (tf * (k1 + 1)) / denominator
    return min(score / 10.0, 1.0)


def _document_frequencies(chunks: list[KnowledgeChunk]) -> dict[str, int]:
    frequencies: dict[str, int] = {}
    for chunk in chunks:
        for token in set(tokenize(chunk.content + " " + chunk.title)):
            frequencies[token] = frequencies.get(token, 0) + 1
    return frequencies

