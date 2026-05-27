from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ragas_loop import RagasEvaluationLoop


def main() -> None:
    parser = argparse.ArgumentParser(description="Show recent lightweight RAGAS evaluation runs.")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    runs = RagasEvaluationLoop().recent_runs(limit=args.limit)
    if not runs:
        print("No RAGAS runs recorded.")
        return
    for run in runs:
        metrics = run["metrics_json"]
        print(f"#{run['id']} thread={run['thread_id']} needs_review={run['needs_review']}")
        print(f"query: {run['query']}")
        print(
            "metrics: "
            f"context_precision={metrics.get('context_precision')}, "
            f"context_recall={metrics.get('context_recall')}, "
            f"response_relevancy={metrics.get('response_relevancy')}, "
            f"faithfulness={metrics.get('faithfulness')}"
        )
        if run.get("review_reason"):
            print(f"review_reason: {run['review_reason']}")
        print()


if __name__ == "__main__":
    main()

