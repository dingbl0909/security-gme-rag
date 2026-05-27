from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class GatewaySettings:
    host: str = "0.0.0.0"
    port: int = 8030
    api_key: str | None = None
    backend: str = "heuristic"
    vllm_base_url: str | None = None
    vllm_model: str = "dots-mocr"
    vllm_prompt_mode: str = "prompt_layout_all_en"
    max_upload_mb: int = 50


def load_settings() -> GatewaySettings:
    return GatewaySettings(
        host=os.getenv("DOTS_GATEWAY_HOST", "0.0.0.0"),
        port=int(os.getenv("DOTS_GATEWAY_PORT", "8030")),
        api_key=os.getenv("DOTS_GATEWAY_API_KEY") or None,
        backend=os.getenv("DOTS_GATEWAY_BACKEND", "heuristic").strip().lower(),
        vllm_base_url=os.getenv("DOTS_VLLM_BASE_URL") or None,
        vllm_model=os.getenv("DOTS_VLLM_MODEL", "dots-mocr"),
        vllm_prompt_mode=os.getenv("DOTS_VLLM_PROMPT_MODE", "prompt_layout_all_en"),
        max_upload_mb=int(os.getenv("DOTS_GATEWAY_MAX_UPLOAD_MB", "50")),
    )
