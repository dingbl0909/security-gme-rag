from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.workflow import SecurityRagWorkflow


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the security multimodal RAG demo.")
    parser.add_argument("query")
    parser.add_argument("--thread-id", default="cli-demo")
    parser.add_argument("--image-ref", default=None)
    args = parser.parse_args()

    response = SecurityRagWorkflow().run(args.query, thread_id=args.thread_id, image_ref=args.image_ref)
    print(response.answer)
    print(f"\n状态：{response.status}")
    if response.workflow_trace:
        print("\n工作流轨迹：")
        for step in response.workflow_trace:
            print(f"- {step}")
    print("\n评估：")
    print(f"- context_precision={response.context_precision}")
    print(f"- context_recall={response.context_recall}")
    print(f"- response_relevancy={response.response_relevancy}")
    print(f"- faithfulness={response.faithfulness}")
    print(f"- needs_review={response.needs_review}")
    if response.review_reason:
        print(f"- review_reason={response.review_reason}")
    if response.status == "interrupted":
        print("\n该请求已中断等待人工审核，可通过 API /resume 继续。")


if __name__ == "__main__":
    main()

