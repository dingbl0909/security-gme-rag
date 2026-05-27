from __future__ import annotations

import mimetypes
import uuid
from pathlib import Path

from fastapi import UploadFile

from app.config import Settings, get_settings
from app.ingest.types import DocumentArtifact


async def save_upload(file: UploadFile, settings: Settings | None = None) -> DocumentArtifact:
    settings = settings or get_settings()
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(file.filename or "upload.bin").suffix.lower() or ".bin"
    artifact_id = f"upload-{uuid.uuid4().hex[:12]}"
    target = settings.uploads_dir / f"{artifact_id}{suffix}"
    payload = await file.read()
    target.write_bytes(payload)
    mime_type = file.content_type or mimetypes.guess_type(target.name)[0] or "application/octet-stream"
    return DocumentArtifact(
        artifact_id=artifact_id,
        uri=str(target.resolve()),
        mime_type=mime_type,
        local_path=str(target),
        metadata={"suffix": suffix, "original_filename": file.filename or target.name},
    )
