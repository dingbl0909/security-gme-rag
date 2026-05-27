from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from app.models import RawDocument


BlockKind = Literal["heading", "paragraph", "table", "image", "list"]


@dataclass(frozen=True)
class DocumentArtifact:
    """A single ingestible object discovered by a document source."""

    artifact_id: str
    uri: str
    mime_type: str
    local_path: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class LayoutBlock:
    """Structured block produced by Dots.OCR or a local layout parser."""

    kind: BlockKind
    text: str
    level: int | None = None
    image_ref: str | None = None
    image_base64: str | None = None
    ocr_text: str | None = None


@dataclass(frozen=True)
class ParsedDocument:
    """Unified output of the parsing stage before chunking."""

    doc_id: str
    title: str
    source: str
    blocks: list[LayoutBlock]
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def content(self) -> str:
        return render_blocks(self.blocks)

    def to_raw_document(self) -> RawDocument:
        return RawDocument(
            doc_id=self.doc_id,
            title=self.title,
            source=self.source,
            content=self.content,
            metadata={**self.metadata, "block_count": str(len(self.blocks))},
        )


def render_blocks(blocks: list[LayoutBlock]) -> str:
    lines: list[str] = []
    for block in blocks:
        if block.kind == "heading" and block.level:
            prefix = "#" * min(block.level, 6)
            lines.append(f"{prefix} {block.text}".strip())
        elif block.kind == "image" and block.image_ref:
            alt = block.text or block.image_ref
            lines.append(f"![{alt}]({block.image_ref})")
            if block.ocr_text:
                lines.append(block.ocr_text)
        elif block.kind == "table":
            lines.append(f"[TABLE]\n{block.text}")
        else:
            if block.text.strip():
                lines.append(block.text.strip())
    return "\n\n".join(lines).strip()
