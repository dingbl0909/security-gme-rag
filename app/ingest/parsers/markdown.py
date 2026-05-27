from __future__ import annotations

import re
from pathlib import Path

from app.ingest.types import DocumentArtifact, LayoutBlock, ParsedDocument


TITLE_RE = re.compile(r"^#\s+(.+)$", re.MULTILINE)
HEADING_RE = re.compile(r"^(?P<hashes>#{1,6})\s+(?P<text>.+)$")
IMAGE_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<path>[^)]+)\)")


class MarkdownLayoutParser:
    """Local parser that preserves headings, paragraphs and image references."""

    def parse(self, artifact: DocumentArtifact) -> ParsedDocument:
        path = Path(artifact.local_path or artifact.uri)
        text = path.read_text(encoding="utf-8")
        title_match = TITLE_RE.search(text)
        title = title_match.group(1).strip() if title_match else path.stem
        blocks = _markdown_to_blocks(text)
        return ParsedDocument(
            doc_id=artifact.artifact_id,
            title=title,
            source=artifact.uri,
            blocks=blocks,
            metadata={"parser": "markdown_layout", **artifact.metadata},
        )


def _markdown_to_blocks(text: str) -> list[LayoutBlock]:
    blocks: list[LayoutBlock] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        heading = HEADING_RE.match(line)
        if heading:
            blocks.append(
                LayoutBlock(
                    kind="heading",
                    text=heading.group("text").strip(),
                    level=len(heading.group("hashes")),
                )
            )
            continue
        for match in IMAGE_RE.finditer(line):
            blocks.append(
                LayoutBlock(
                    kind="image",
                    text=match.group("alt").strip() or Path(match.group("path")).name,
                    image_ref=_normalize_image_ref(match.group("path")),
                )
            )
        remainder = IMAGE_RE.sub("", line).strip()
        if remainder:
            blocks.append(LayoutBlock(kind="paragraph", text=remainder))
    return blocks


def _normalize_image_ref(image_path: str) -> str:
    return Path(image_path).name
