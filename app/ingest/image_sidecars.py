from __future__ import annotations

import hashlib
import re
from pathlib import Path

from app.models import KnowledgeChunk


def build_image_sidecar_chunks(images_dir: Path) -> list[KnowledgeChunk]:
    chunks: list[KnowledgeChunk] = []
    for path in sorted(images_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        title = _extract_field(text, "description") or path.stem
        description = _extract_field(text, "description") or text
        ocr_text = _extract_field(text, "ocr_text")
        content = f"图片语义描述：{description}"
        if ocr_text:
            content += f"\nOCR：{ocr_text}"
        digest = hashlib.sha1(f"image:{path.name}:{content}".encode("utf-8")).hexdigest()[:16]
        chunks.append(
            KnowledgeChunk(
                chunk_id=f"chunk-{digest}",
                doc_id=f"image-{path.stem}",
                title=title,
                source=str(path.resolve()),
                modality="image",
                content=content,
                image_ref=path.name,
                metadata={"chunker": "image_sidecar"},
            )
        )
    return chunks


def _extract_field(text: str, field: str) -> str | None:
    match = re.search(rf"^{field}:\s*(.+)$", text, flags=re.MULTILINE)
    return match.group(1).strip() if match else None
