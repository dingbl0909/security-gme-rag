from __future__ import annotations

import base64
import json
import urllib.request
from pathlib import Path

from app.config import Settings
from app.ingest.parsers.markdown import MarkdownLayoutParser
from app.ingest.types import DocumentArtifact, LayoutBlock, ParsedDocument


class DotsOcrParser:
    """Call local dots-ocr-gateway; fall back to markdown layout parser when gateway is off."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._markdown = MarkdownLayoutParser()

    def parse(self, artifact: DocumentArtifact) -> ParsedDocument:
        if self.settings.parser_provider == "dots_ocr" and self.settings.dots_ocr_base_url:
            return self._parse_remote(artifact)
        suffix = artifact.metadata.get("suffix", Path(artifact.uri).suffix.lower())
        if suffix in {".md", ".markdown", ".txt"}:
            parsed = self._markdown.parse(artifact)
            return ParsedDocument(
                doc_id=parsed.doc_id,
                title=parsed.title,
                source=parsed.source,
                blocks=parsed.blocks,
                metadata={**parsed.metadata, "parser": "markdown_layout"},
            )
        return self._parse_local_mock(artifact)

    def _parse_remote(self, artifact: DocumentArtifact) -> ParsedDocument:
        if not self.settings.dots_ocr_base_url:
            raise RuntimeError("DOTS_OCR_BASE_URL is required when GME_RAG_PARSER_PROVIDER=dots_ocr")
        path = Path(artifact.local_path or artifact.uri)
        payload = {
            "doc_id": artifact.artifact_id,
            "filename": path.name,
            "mime_type": artifact.mime_type,
            "content_base64": base64.b64encode(path.read_bytes()).decode("ascii"),
        }
        endpoint = self.settings.dots_ocr_base_url.rstrip("/") + self.settings.dots_ocr_parse_path
        headers = {"Content-Type": "application/json"}
        if self.settings.dots_ocr_api_key:
            headers["Authorization"] = f"Bearer {self.settings.dots_ocr_api_key}"
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            data = json.loads(response.read().decode("utf-8"))
        return _parsed_from_dots_response(artifact, data)

    def _parse_local_mock(self, artifact: DocumentArtifact) -> ParsedDocument:
        path = Path(artifact.local_path or artifact.uri)
        suffix = path.suffix.lower()
        title = path.stem
        blocks = [
            LayoutBlock(
                kind="paragraph",
                text=(
                    f"[demo] 未配置 Dots.OCR 网关，已为 {suffix} 生成占位解析。"
                    "请启动 services/dots_ocr_gateway 并设置 DOTS_OCR_BASE_URL。"
                ),
            )
        ]
        return ParsedDocument(
            doc_id=artifact.artifact_id,
            title=title,
            source=artifact.uri,
            blocks=blocks,
            metadata={"parser": "dots_ocr_local_mock", **artifact.metadata},
        )


def _parsed_from_dots_response(artifact: DocumentArtifact, data: dict) -> ParsedDocument:
    title = data.get("title") or artifact.artifact_id
    raw_blocks = data.get("blocks") or data.get("layout_blocks") or []
    blocks: list[LayoutBlock] = []
    for item in raw_blocks:
        blocks.append(
            LayoutBlock(
                kind=item.get("kind", "paragraph"),  # type: ignore[arg-type]
                text=item.get("text", ""),
                level=item.get("level"),
                image_ref=item.get("image_ref"),
                image_base64=item.get("image_base64"),
                ocr_text=item.get("ocr_text"),
            )
        )
    if not blocks and data.get("content"):
        blocks = [LayoutBlock(kind="paragraph", text=str(data["content"]))]
    parser_name = data.get("parser", "dots_ocr_remote")
    return ParsedDocument(
        doc_id=artifact.artifact_id,
        title=str(title),
        source=artifact.uri,
        blocks=blocks,
        metadata={"parser": parser_name, **artifact.metadata},
    )
