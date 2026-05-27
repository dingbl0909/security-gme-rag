from __future__ import annotations

from fastapi import UploadFile

from app.config import get_settings
from app.ingest.pipeline import IngestionPipeline
from app.ingest.types import LayoutBlock, ParsedDocument
from app.ingest.upload import save_upload
from app.models import DocumentUploadResponse, LayoutBlockOut


async def handle_document_upload(file: UploadFile, *, index: bool = False) -> DocumentUploadResponse:
    settings = get_settings()
    artifact = await save_upload(file, settings)
    pipeline = IngestionPipeline(settings=settings)
    parsed, chunks = pipeline.ingest_artifact(artifact, index=index)
    return _to_response(parsed, chunks, index=index)


def _to_response(parsed: ParsedDocument, chunks: list, *, index: bool) -> DocumentUploadResponse:
    parser = parsed.metadata.get("parser", "unknown")
    return DocumentUploadResponse(
        doc_id=parsed.doc_id,
        title=parsed.title,
        source=parsed.source,
        parser=parser,
        block_count=len(parsed.blocks),
        blocks=[_block_out(block) for block in parsed.blocks],
        content_preview=parsed.content[:500],
        indexed_chunks=len(chunks) if index else None,
    )


def _block_out(block: LayoutBlock) -> LayoutBlockOut:
    return LayoutBlockOut(
        kind=block.kind,
        text=block.text,
        level=block.level,
        image_ref=block.image_ref,
        has_image_base64=bool(block.image_base64),
        ocr_text=block.ocr_text,
    )
