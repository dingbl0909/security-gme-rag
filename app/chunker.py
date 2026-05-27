from __future__ import annotations

import hashlib
import re
from pathlib import Path

from app.models import KnowledgeChunk, RawDocument


IMAGE_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<path>[^)]+)\)")


def chunk_documents(documents: list[RawDocument], max_chars: int = 420) -> list[KnowledgeChunk]:
    chunks: list[KnowledgeChunk] = []
    for document in documents:
        sections = _split_sections(document.content)
        for section in sections:
            chunks.extend(_chunk_section(document, section, max_chars=max_chars))
    return chunks


def _split_sections(content: str) -> list[str]:
    lines = content.splitlines()
    sections: list[list[str]] = []
    current: list[str] = []
    for line in lines:
        if line.startswith("## ") and current:
            sections.append(current)
            current = [line]
        else:
            current.append(line)
    if current:
        sections.append(current)
    return ["\n".join(section).strip() for section in sections if "\n".join(section).strip()]


def _chunk_section(document: RawDocument, section: str, max_chars: int) -> list[KnowledgeChunk]:
    chunks: list[KnowledgeChunk] = []
    image_matches = list(IMAGE_RE.finditer(section))
    if image_matches:
        for match in image_matches:
            image_ref = _normalize_image_ref(match.group("path"))
            image_text = _load_image_text(document.source, image_ref)
            content = f"{section}\n\n图片语义描述：{match.group('alt')}。{image_text}".strip()
            chunks.append(_build_chunk(document, content, "text_image", image_ref=image_ref))
    text_without_images = IMAGE_RE.sub("", section).strip()
    for part in _semantic_windows(text_without_images, max_chars=max_chars):
        chunks.append(_build_chunk(document, part, "text"))
    return chunks


def _semantic_windows(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text] if text else []
    sentences = re.split(r"(?<=[。！？\n])", text)
    windows: list[str] = []
    current = ""
    for sentence in sentences:
        if len(current) + len(sentence) > max_chars and current:
            windows.append(current.strip())
            current = sentence
        else:
            current += sentence
    if current.strip():
        windows.append(current.strip())
    return windows


def _build_chunk(document: RawDocument, content: str, modality: str, image_ref: str | None = None) -> KnowledgeChunk:
    digest = hashlib.sha1(f"{document.doc_id}:{modality}:{image_ref}:{content}".encode("utf-8")).hexdigest()[:16]
    return KnowledgeChunk(
        chunk_id=f"chunk-{digest}",
        doc_id=document.doc_id,
        title=document.title,
        source=document.source,
        modality=modality,  # type: ignore[arg-type]
        content=content,
        image_ref=image_ref,
        metadata={"chunker": "section_semantic_window"},
    )


def _normalize_image_ref(image_path: str) -> str:
    return Path(image_path).name


def _load_image_text(document_source: str, image_ref: str) -> str:
    docs_dir = Path(document_source).parent
    candidate = docs_dir.parent / "images" / image_ref
    if candidate.exists():
        return candidate.read_text(encoding="utf-8")
    return f"image_ref={image_ref}"

