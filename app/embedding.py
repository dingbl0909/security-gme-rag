from __future__ import annotations

import hashlib
import json
import math
import urllib.request

from app.config import Settings, get_settings


class GmeLikeEmbedder:
    """Deterministic local substitute for a GME multimodal embedder."""

    def __init__(self, dimensions: int = 64):
        self.dimensions = dimensions

    def embed(self, text: str, image_ref: str | None = None) -> list[float]:
        seed = f"{image_ref or ''}\n{text}"
        vector = [0.0] * self.dimensions
        for token in _tokens(seed):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        return _normalize(vector)


class GmeApiEmbedder:
    """HTTP adapter for a private GME Qwen2-VL embedding service.

    The adapter expects an OpenAI-compatible embedding response:
    {"data": [{"embedding": [...]}]}.
    """

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        if not self.settings.gme_api_key:
            raise RuntimeError("GME_API_KEY is required when GME_RAG_EMBEDDING_PROVIDER=gme")
        if not self.settings.gme_base_url:
            raise RuntimeError("GME_BASE_URL is required when GME_RAG_EMBEDDING_PROVIDER=gme")
        self.endpoint = self.settings.gme_base_url.rstrip("/") + self.settings.gme_embedding_path

    def embed(self, text: str, image_ref: str | None = None) -> list[float]:
        payload = {
            "model": self.settings.gme_model,
            "input": [{"text": text, "image": image_ref} if image_ref else text],
        }
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.settings.gme_api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            data = json.loads(response.read().decode("utf-8"))
        embedding = _extract_embedding(data)
        return _normalize([float(value) for value in embedding])


def build_embedder(settings: Settings | None = None):
    settings = settings or get_settings()
    if settings.embedding_provider == "local":
        return GmeLikeEmbedder()
    if settings.embedding_provider == "gme":
        return GmeApiEmbedder(settings)
    raise ValueError(f"Unsupported embedding provider: {settings.embedding_provider}")


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    return sum(a * b for a, b in zip(left, right))


def _extract_embedding(data: dict) -> list[float]:
    if "data" in data and data["data"]:
        first = data["data"][0]
        if isinstance(first, dict) and "embedding" in first:
            return first["embedding"]
    if "embedding" in data:
        return data["embedding"]
    raise RuntimeError("GME embedding response does not contain an embedding vector")


def _tokens(text: str) -> list[str]:
    lowered = text.lower()
    tokens = []
    buffer = ""
    for char in lowered:
        if char.isalnum() or "\u4e00" <= char <= "\u9fff":
            buffer += char
        else:
            if buffer:
                tokens.append(buffer)
                buffer = ""
    if buffer:
        tokens.append(buffer)
    tokens.extend([char for char in lowered if "\u4e00" <= char <= "\u9fff"])
    return tokens


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]

