from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from app.config import get_settings
from app.models import KnowledgeChunk


class RagDatabase:
    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or get_settings().db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def init_schema(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    source TEXT NOT NULL,
                    modality TEXT NOT NULL,
                    content TEXT NOT NULL,
                    image_ref TEXT,
                    metadata_json TEXT NOT NULL,
                    vector_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS memories (
                    thread_id TEXT PRIMARY KEY,
                    summary TEXT NOT NULL DEFAULT '',
                    facts_json TEXT NOT NULL DEFAULT '[]',
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS ragas_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    thread_id TEXT NOT NULL,
                    query TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    evidence_json TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    needs_review INTEGER NOT NULL,
                    review_reason TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                """
            )

    def clear_chunks(self) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM chunks")

    def upsert_chunk(self, chunk: KnowledgeChunk, vector: list[float]) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO chunks(chunk_id, doc_id, title, source, modality, content, image_ref, metadata_json, vector_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(chunk_id) DO UPDATE SET
                    title=excluded.title,
                    source=excluded.source,
                    modality=excluded.modality,
                    content=excluded.content,
                    image_ref=excluded.image_ref,
                    metadata_json=excluded.metadata_json,
                    vector_json=excluded.vector_json
                """,
                (
                    chunk.chunk_id,
                    chunk.doc_id,
                    chunk.title,
                    chunk.source,
                    chunk.modality,
                    chunk.content,
                    chunk.image_ref,
                    json.dumps(chunk.metadata, ensure_ascii=False),
                    json.dumps(vector),
                ),
            )

    def load_chunks(self) -> list[tuple[KnowledgeChunk, list[float]]]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM chunks").fetchall()
        result = []
        for row in rows:
            chunk = KnowledgeChunk(
                chunk_id=row["chunk_id"],
                doc_id=row["doc_id"],
                title=row["title"],
                source=row["source"],
                modality=row["modality"],
                content=row["content"],
                image_ref=row["image_ref"],
                metadata=json.loads(row["metadata_json"]),
            )
            result.append((chunk, json.loads(row["vector_json"])))
        return result

    def get_memory(self, thread_id: str) -> tuple[str, list[str]]:
        with self.connect() as conn:
            row = conn.execute("SELECT summary, facts_json FROM memories WHERE thread_id=?", (thread_id,)).fetchone()
        if row is None:
            return "", []
        return row["summary"], json.loads(row["facts_json"])

    def save_memory(self, thread_id: str, summary: str, facts: list[str]) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO memories(thread_id, summary, facts_json, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(thread_id) DO UPDATE SET
                    summary=excluded.summary,
                    facts_json=excluded.facts_json,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (thread_id, summary, json.dumps(facts, ensure_ascii=False)),
            )

    def save_ragas_run(
        self,
        thread_id: str,
        query: str,
        answer: str,
        evidence: list[dict],
        metrics: dict[str, float],
        needs_review: bool,
        review_reason: str | None,
    ) -> int:
        with self.connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO ragas_runs(thread_id, query, answer, evidence_json, metrics_json, needs_review, review_reason)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    thread_id,
                    query,
                    answer,
                    json.dumps(evidence, ensure_ascii=False),
                    json.dumps(metrics, ensure_ascii=False),
                    1 if needs_review else 0,
                    review_reason,
                ),
            )
            return int(cursor.lastrowid)

    def list_ragas_runs(self, limit: int = 20) -> list[dict]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM ragas_runs ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["evidence_json"] = json.loads(item["evidence_json"])
            item["metrics_json"] = json.loads(item["metrics_json"])
            item["needs_review"] = bool(item["needs_review"])
            result.append(item)
        return result

