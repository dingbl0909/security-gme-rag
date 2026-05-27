from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RagasMetrics:
    context_precision: float
    context_recall: float
    response_relevancy: float
    faithfulness: float

