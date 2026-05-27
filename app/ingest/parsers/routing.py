from __future__ import annotations

from app.config import Settings
from app.ingest.parsers.dots_ocr import DotsOcrParser
from app.ingest.parsers.markdown import MarkdownLayoutParser
from app.ingest.types import DocumentArtifact, ParsedDocument

MARKDOWN_SUFFIXES = {".md", ".markdown", ".txt"}


class RoutingDocumentParser:
    """样例 Markdown 走本地版面解析；上传的 PDF/图片等走 Dots.OCR 网关。"""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._markdown = MarkdownLayoutParser()
        self._dots = DotsOcrParser(settings)

    def parse(self, artifact: DocumentArtifact) -> ParsedDocument:
        suffix = artifact.metadata.get("suffix", "")
        if suffix in MARKDOWN_SUFFIXES:
            return self._markdown.parse(artifact)
        return self._dots.parse(artifact)
