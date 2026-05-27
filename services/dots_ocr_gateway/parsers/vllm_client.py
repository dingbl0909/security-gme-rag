from __future__ import annotations

import base64
import json
import urllib.request
from typing import Any

from config import GatewaySettings
from parsers.heuristic import parse_bytes


def parse_with_vllm(
    settings: GatewaySettings,
    filename: str,
    mime_type: str,
    payload: bytes,
) -> dict[str, Any]:
    if not settings.vllm_base_url:
        raise RuntimeError("DOTS_VLLM_BASE_URL is required when DOTS_GATEWAY_BACKEND=vllm")
    image_b64 = base64.b64encode(payload).decode("ascii")
    prompt = (
        f"Parse this document with layout mode {settings.vllm_prompt_mode}. "
        "Return JSON with fields: title, blocks[]. Each block has kind, text, optional level, "
        "image_ref, ocr_text."
    )
    body = {
        "model": settings.vllm_model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{image_b64}"},
                    },
                ],
            }
        ],
        "temperature": 0.1,
    }
    endpoint = settings.vllm_base_url.rstrip("/") + "/v1/chat/completions"
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            data = json.loads(response.read().decode("utf-8"))
        content = data["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        parsed["parser"] = "dots_ocr_gateway_vllm"
        return parsed
    except Exception:
        fallback = parse_bytes(filename, mime_type, payload)
        fallback["parser"] = "dots_ocr_gateway_vllm_fallback_heuristic"
        return fallback
