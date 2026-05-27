from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    db_path: Path = PROJECT_ROOT / "data/db/security_gme_rag.sqlite3"
    docs_dir: Path = PROJECT_ROOT / "data/security_docs"
    uploads_dir: Path = PROJECT_ROOT / "data/uploads"
    images_dir: Path = PROJECT_ROOT / "data/images"
    dense_weight: float = 0.6
    sparse_weight: float = 0.4
    top_k: int = 5
    review_threshold: float = 0.28
    embedding_provider: str = "local"
    gme_api_key: str | None = None
    gme_base_url: str | None = None
    gme_embedding_path: str = "/embeddings"
    gme_model: str = "iic/gme-Qwen2-VL-7B-Instruct"
    gme_dimensions: int = 1024
    memory_provider: str = "sqlite"
    redis_url: str = "redis://localhost:6379/0"
    redis_prefix: str = "security-gme-rag"
    vector_store: str = "sqlite"
    milvus_uri: str = "http://localhost:19530"
    milvus_token: str | None = None
    milvus_collection: str = "security_gme_rag_chunks"
    parser_provider: str = "local"
    dots_ocr_api_key: str | None = None
    dots_ocr_base_url: str | None = None
    dots_ocr_parse_path: str = "/parse"

    def ensure_dirs(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    if load_dotenv is not None:
        load_dotenv(PROJECT_ROOT / ".env", override=False)
    settings = Settings(
        embedding_provider=os.getenv("GME_RAG_EMBEDDING_PROVIDER", "local").strip().lower(),
        gme_api_key=os.getenv("GME_API_KEY") or None,
        gme_base_url=os.getenv("GME_BASE_URL") or None,
        gme_embedding_path=os.getenv("GME_EMBEDDING_PATH", "/embeddings"),
        gme_model=os.getenv("GME_MODEL", "iic/gme-Qwen2-VL-7B-Instruct"),
        gme_dimensions=int(os.getenv("GME_DIMENSIONS", "1024")),
        memory_provider=os.getenv("GME_RAG_MEMORY_PROVIDER", "sqlite").strip().lower(),
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
        redis_prefix=os.getenv("REDIS_PREFIX", "security-gme-rag"),
        vector_store=os.getenv("GME_RAG_VECTOR_STORE", "sqlite").strip().lower(),
        milvus_uri=os.getenv("MILVUS_URI", "http://localhost:19530"),
        milvus_token=os.getenv("MILVUS_TOKEN") or None,
        milvus_collection=os.getenv("MILVUS_COLLECTION", "security_gme_rag_chunks"),
        parser_provider=os.getenv("GME_RAG_PARSER_PROVIDER", "local").strip().lower(),
        dots_ocr_api_key=os.getenv("DOTS_OCR_API_KEY") or None,
        dots_ocr_base_url=os.getenv("DOTS_OCR_BASE_URL") or None,
        dots_ocr_parse_path=os.getenv("DOTS_OCR_PARSE_PATH", "/parse"),
    )
    settings.ensure_dirs()
    return settings

