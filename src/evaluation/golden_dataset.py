from __future__ import annotations

import json
from pathlib import Path

from src.settings import ROOT_DIR


DEFAULT_GOLDEN_PATH = ROOT_DIR / "data" / "qa_dataset" / "golden_dataset.json"


def load_golden_dataset(path: str | Path | None = None) -> list[dict]:
    dataset_path = Path(path) if path else DEFAULT_GOLDEN_PATH
    if not dataset_path.exists():
        return []
    return json.loads(dataset_path.read_text(encoding="utf-8"))

