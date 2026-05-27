from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field


Modality = Literal["text", "image", "text_image"]


@dataclass(frozen=True)
class RawDocument:
    doc_id: str
    title: str
    source: str
    content: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class KnowledgeChunk:
    chunk_id: str
    doc_id: str
    title: str
    source: str
    modality: Modality
    content: str
    image_ref: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalHit:
    chunk: KnowledgeChunk
    dense_score: float
    sparse_score: float
    hybrid_score: float


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    thread_id: str = "default-thread"
    image_ref: str | None = None
    top_k: int | None = None


class EvidenceItem(BaseModel):
    title: str
    source: str
    modality: str
    snippet: str
    image_ref: str | None = None
    dense_score: float
    sparse_score: float
    hybrid_score: float


class QueryResponse(BaseModel):
    answer: str
    thread_id: str
    intent: str
    evidence: list[EvidenceItem]
    context_summary: str = ""
    context_precision: float
    context_recall: float
    response_relevancy: float = 0.0
    faithfulness: float
    needs_review: bool
    review_reason: str | None = None
    status: str = "completed"
    workflow_trace: list[str] = Field(default_factory=list)
    review_payload: dict[str, Any] | None = None


class ResumeRequest(BaseModel):
    thread_id: str
    approve: bool
    reviewer_note: str = ""


class LayoutBlockOut(BaseModel):
    kind: str
    text: str
    level: int | None = None
    image_ref: str | None = None
    has_image_base64: bool = False
    ocr_text: str | None = None


class DocumentUploadResponse(BaseModel):
    doc_id: str
    title: str
    source: str
    parser: str
    block_count: int
    blocks: list[LayoutBlockOut]
    content_preview: str
    indexed_chunks: int | None = None


