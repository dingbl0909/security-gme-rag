#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

GATEWAY = "http://127.0.0.1:8030"
RAG = "http://127.0.0.1:8020"


def _get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def _post_json(url: str, payload: dict) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def test_offline_pipeline() -> None:
    from app.ingestion import ingest_demo_corpus

    count = ingest_demo_corpus()
    assert count > 0, "ingest should create chunks"
    print(f"[ok] offline ingest indexed {count} chunks")


def test_http_services() -> None:
    gateway = _get(f"{GATEWAY}/health")
    assert gateway["status"] == "ok"
    print(f"[ok] gateway health backend={gateway['backend']}")

    rag = _get(f"{RAG}/health")
    assert rag["status"] == "ok"
    print(f"[ok] rag health parser={rag['parser_provider']}")

    ingest = _post_json(f"{RAG}/ingest", {})
    assert ingest["indexed_chunks"] > 0
    print(f"[ok] rag ingest chunks={ingest['indexed_chunks']}")

    query = _post_json(
        f"{RAG}/query",
        {"query": "摄像头离线如何排查？", "thread_id": "smoke-test"},
    )
    assert query.get("answer"), "query should return answer"
    print(f"[ok] rag query intent={query.get('intent')}")


def main() -> None:
    test_offline_pipeline()
    try:
        test_http_services()
    except urllib.error.URLError as exc:
        print(f"[skip] HTTP 服务未启动: {exc}")
        print("可先运行: bash scripts/start_local.sh")
        return
    print("smoke test passed")


if __name__ == "__main__":
    main()
