from __future__ import annotations

import base64
import re
from html.parser import HTMLParser
from io import BytesIO
from pathlib import Path
from typing import Any


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        text = data.strip()
        if text:
            self.parts.append(text)


def parse_bytes(filename: str, mime_type: str, payload: bytes) -> dict[str, Any]:
    suffix = Path(filename).suffix.lower()
    if suffix in {".md", ".markdown", ".txt"} or mime_type.startswith("text/"):
        return _parse_text(filename, payload.decode("utf-8", errors="ignore"))
    if suffix == ".pdf" or mime_type == "application/pdf":
        return _parse_pdf(filename, payload)
    if suffix in {".png", ".jpg", ".jpeg", ".webp", ".bmp"} or mime_type.startswith("image/"):
        return _parse_image(filename, payload)
    if suffix in {".html", ".htm"}:
        return _parse_html(filename, payload.decode("utf-8", errors="ignore"))
    raise ValueError(f"不支持的文件类型: {suffix or mime_type}")


def _parse_text(filename: str, text: str) -> dict[str, Any]:
    blocks: list[dict[str, Any]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        heading = re.match(r"^(#{1,6})\s+(.+)$", stripped)
        if heading:
            blocks.append(
                {
                    "kind": "heading",
                    "text": heading.group(2).strip(),
                    "level": len(heading.group(1)),
                }
            )
            continue
        image = re.match(r"!\[(.*?)\]\((.*?)\)", stripped)
        if image:
            blocks.append(
                {
                    "kind": "image",
                    "text": image.group(1) or Path(image.group(2)).name,
                    "image_ref": Path(image.group(2)).name,
                }
            )
            continue
        blocks.append({"kind": "paragraph", "text": stripped})
    return _response(filename, blocks)


def _parse_html(filename: str, html: str) -> dict[str, Any]:
    extractor = _HTMLTextExtractor()
    extractor.feed(html)
    blocks = [{"kind": "paragraph", "text": part} for part in extractor.parts]
    return _response(filename, blocks)


def _parse_pdf(filename: str, payload: bytes) -> dict[str, Any]:
    try:
        import fitz  # pymupdf
    except ImportError as exc:
        raise RuntimeError("解析 PDF 需要安装 pymupdf: pip install pymupdf") from exc

    blocks: list[dict[str, Any]] = []
    with fitz.open(stream=payload, filetype="pdf") as doc:
        for page_index, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            if not text:
                continue
            blocks.append(
                {
                    "kind": "heading",
                    "text": f"第 {page_index} 页",
                    "level": 2,
                }
            )
            for paragraph in _split_paragraphs(text):
                blocks.append({"kind": "paragraph", "text": paragraph})
            for image in page.get_images(full=True):
                xref = image[0]
                try:
                    extracted = doc.extract_image(xref)
                    image_bytes = extracted["image"]
                    blocks.append(
                        {
                            "kind": "image",
                            "text": f"{filename}-p{page_index}-img{xref}",
                            "image_ref": f"{Path(filename).stem}-p{page_index}-{xref}.png",
                            "image_base64": base64.b64encode(image_bytes).decode("ascii"),
                        }
                    )
                except Exception:
                    continue
    return _response(filename, blocks)


def _parse_image(filename: str, payload: bytes) -> dict[str, Any]:
    ocr_text = _try_ocr(payload)
    blocks = [
        {
            "kind": "image",
            "text": Path(filename).stem,
            "image_ref": Path(filename).name,
            "image_base64": base64.b64encode(payload).decode("ascii"),
            "ocr_text": ocr_text,
        }
    ]
    if ocr_text:
        blocks.append({"kind": "paragraph", "text": ocr_text})
    return _response(filename, blocks)


def _try_ocr(payload: bytes) -> str | None:
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        return None
    try:
        image = Image.open(BytesIO(payload))
        text = pytesseract.image_to_string(image, lang="chi_sim+eng").strip()
        return text or None
    except Exception:
        return None


def _split_paragraphs(text: str) -> list[str]:
    parts = [part.strip() for part in re.split(r"\n{2,}", text) if part.strip()]
    if parts:
        return parts
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return [" ".join(lines)] if lines else []


def _response(filename: str, blocks: list[dict[str, Any]]) -> dict[str, Any]:
    title = Path(filename).stem
    if blocks and blocks[0]["kind"] == "heading":
        title = str(blocks[0]["text"])
    return {
        "title": title,
        "blocks": blocks,
        "parser": "dots_ocr_gateway_heuristic",
    }
