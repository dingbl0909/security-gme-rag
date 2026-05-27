from __future__ import annotations

from fastapi import FastAPI, File, Query, UploadFile

from app import __version__
from app.config import get_settings
from app.documents import handle_document_upload
from app.ingestion import ingest_demo_corpus_with_report
from app.models import DocumentUploadResponse, QueryRequest, QueryResponse, ResumeRequest
from app.ragas_loop import RagasEvaluationLoop
from app.workflow import SecurityRagWorkflow


app = FastAPI(title="security-gme-rag", version=__version__)
workflow = SecurityRagWorkflow()
ragas_loop = RagasEvaluationLoop()


@app.get("/health")
def health() -> dict[str, str | bool]:
    settings = get_settings()
    return {
        "status": "ok",
        "app": "security-gme-rag",
        "version": __version__,
        "parser_provider": settings.parser_provider,
        "dots_ocr_configured": bool(settings.dots_ocr_base_url),
    }


@app.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    index: bool = Query(False, description="解析后是否写入知识库索引"),
) -> DocumentUploadResponse:
    """上传 PDF/图片/Office/HTML 等非结构化资料，经 Dots.OCR（或本地回退）解析为版面块。"""
    return await handle_document_upload(file, index=index)


@app.post("/ingest")
def ingest() -> dict[str, int | list[str]]:
    report = ingest_demo_corpus_with_report()
    return {
        "artifact_count": report.artifact_count,
        "parsed_count": report.parsed_count,
        "indexed_chunks": report.chunk_count,
        "skipped": report.skipped,
    }


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    return workflow.run(
        query=request.query,
        thread_id=request.thread_id,
        image_ref=request.image_ref,
        top_k=request.top_k,
    )


@app.post("/resume", response_model=QueryResponse)
def resume(request: ResumeRequest) -> QueryResponse:
    return workflow.resume(
        thread_id=request.thread_id,
        approve=request.approve,
        reviewer_note=request.reviewer_note,
    )


@app.get("/ragas/runs")
def ragas_runs(limit: int = 20) -> list[dict]:
    return ragas_loop.recent_runs(limit=limit)

