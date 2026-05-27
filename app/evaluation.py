from __future__ import annotations

from app.models import RetrievalHit
from app.retrieval import tokenize


def context_precision(query: str, hits: list[RetrievalHit]) -> float:
    if not hits:
        return 0.0
    query_tokens = set(tokenize(query))
    relevant = 0
    for hit in hits:
        content_tokens = set(tokenize(hit.chunk.content))
        if query_tokens & content_tokens:
            relevant += 1
    return round(relevant / len(hits), 4)


def context_recall(query: str, hits: list[RetrievalHit]) -> float:
    query_tokens = set(tokenize(query))
    if not query_tokens:
        return 0.0
    covered: set[str] = set()
    for hit in hits:
        covered |= query_tokens & set(tokenize(hit.chunk.content))
    return round(len(covered) / len(query_tokens), 4)


def faithfulness(answer: str, hits: list[RetrievalHit]) -> float:
    if not hits:
        return 0.0
    evidence_text = " ".join(hit.chunk.content for hit in hits)
    answer_tokens = set(tokenize(answer))
    evidence_tokens = set(tokenize(evidence_text))
    if not answer_tokens:
        return 0.0
    return round(len(answer_tokens & evidence_tokens) / len(answer_tokens), 4)


def response_relevancy(query: str, answer: str) -> float:
    query_tokens = set(tokenize(query))
    answer_tokens = set(tokenize(answer))
    if not query_tokens or not answer_tokens:
        return 0.0
    overlap = len(query_tokens & answer_tokens) / len(query_tokens)
    coverage = len(query_tokens & answer_tokens) / max(len(answer_tokens), 1)
    # Favor answers that cover query terms without becoming pure keyword dumps.
    return round(0.75 * overlap + 0.25 * coverage, 4)

