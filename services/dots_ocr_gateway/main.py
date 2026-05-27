from __future__ import annotations

import base64
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from config import GatewaySettings, load_settings
from parsers.heuristic import parse_bytes
from parsers.vllm_client import parse_with_vllm


app = FastAPI(title="dots-ocr-gateway", version="0.1.0")
SETTINGS = load_settings()


class ParseRequest(BaseModel):
    doc_id: str | None = None
    filename: str
    mime_type: str = "application/octet-stream"
    content_base64: str = Field(..., min_length=1)


def _verify_api_key(
    settings: GatewaySettings,
    authorization: str | None = Header(default=None),
) -> None:
    if not settings.api_key:
        return
    if not authorization or authorization != f"Bearer {settings.api_key}":
        raise HTTPException(status_code=401, detail="Invalid API key")


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "dots-ocr-gateway",
        "backend": SETTINGS.backend,
        "vllm_configured": bool(SETTINGS.vllm_base_url),
    }


@app.post("/parse")
def parse_document(
    request: ParseRequest,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    _verify_api_key(SETTINGS, authorization)
    try:
        payload = base64.b64decode(request.content_base64, validate=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid content_base64") from exc
    max_bytes = SETTINGS.max_upload_mb * 1024 * 1024
    if len(payload) > max_bytes:
        raise HTTPException(status_code=413, detail=f"File exceeds {SETTINGS.max_upload_mb}MB")

    filename = request.filename
    mime_type = request.mime_type
    try:
        if SETTINGS.backend == "vllm":
            result = parse_with_vllm(SETTINGS, filename, mime_type, payload)
        else:
            result = parse_bytes(filename, mime_type, payload)
    except ValueError as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    result["doc_id"] = request.doc_id
    return result
