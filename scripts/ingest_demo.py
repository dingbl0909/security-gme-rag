from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ingestion import ingest_demo_corpus


def main() -> None:
    count = ingest_demo_corpus()
    print(f"Indexed {count} multimodal knowledge chunks.")


if __name__ == "__main__":
    main()

