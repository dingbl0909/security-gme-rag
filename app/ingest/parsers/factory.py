from __future__ import annotations

from app.config import Settings, get_settings
from app.ingest.parsers.base import DocumentParser
from app.ingest.parsers.dots_ocr import DotsOcrParser
from app.ingest.parsers.markdown import MarkdownLayoutParser
from app.ingest.parsers.routing import RoutingDocumentParser


def build_document_parser(settings: Settings | None = None) -> DocumentParser:
    settings = settings or get_settings()
    if settings.parser_provider == "markdown":
        return MarkdownLayoutParser()
    if settings.parser_provider == "local" and not settings.dots_ocr_base_url:
        return DotsOcrParser(settings)
    if settings.parser_provider in {"dots_ocr", "local"}:
        return RoutingDocumentParser(settings)
    raise ValueError(f"Unsupported parser provider: {settings.parser_provider}")
