from __future__ import annotations

import json

from app.config import get_settings
from app.database import RagDatabase


class MemoryStore:
    def __init__(self, database: RagDatabase | None = None):
        self.settings = get_settings()
        self.database = database or RagDatabase()
        self.database.init_schema()
        self.redis = self._connect_redis() if self.settings.memory_provider == "redis" else None

    def load(self, thread_id: str) -> tuple[str, list[str]]:
        if self.redis is not None:
            payload = self.redis.get(self._key(thread_id))
            if payload:
                data = json.loads(payload)
                return data.get("summary", ""), data.get("facts", [])
        return self.database.get_memory(thread_id)

    def save_turn(self, thread_id: str, query: str, answer: str, facts: list[str]) -> None:
        summary = f"最近问题：{query}\n最近回答：{answer[:240]}"
        existing_summary, existing_facts = self.load(thread_id)
        merged_facts = list(dict.fromkeys(existing_facts + facts))
        if existing_summary:
            summary = existing_summary[-300:] + "\n" + summary
        summary = summary[-600:]
        merged_facts = merged_facts[-20:]
        if self.redis is not None:
            self.redis.set(
                self._key(thread_id),
                json.dumps({"summary": summary, "facts": merged_facts}, ensure_ascii=False),
                ex=60 * 60 * 24,
            )
        self.database.save_memory(thread_id, summary, merged_facts)

    def _connect_redis(self):
        try:
            import redis
        except ImportError as exc:  # pragma: no cover - optional production dependency
            raise RuntimeError("redis is required when GME_RAG_MEMORY_PROVIDER=redis") from exc
        return redis.Redis.from_url(self.settings.redis_url, decode_responses=True)

    def _key(self, thread_id: str) -> str:
        return f"{self.settings.redis_prefix}:short_memory:{thread_id}"

