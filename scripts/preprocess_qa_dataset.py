from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from src.database.models import QAPair
from src.database.mongo_client import MongoService
from src.processing.qa_preprocessor import load_law_signals
from src.processing.qa_preprocessor import preprocess_qa_items
from src.settings import ROOT_DIR

RAW_QA_DIR = ROOT_DIR / "data" / "raw" / "qa_pairs"
RAW_INPUT_FILE = RAW_QA_DIR / "qa_pairs_raw.json"
PROCESSED_OUTPUT_FILE = ROOT_DIR / "data" / "processed" / "qa_pairs" / "qa_pairs_processed.json"
LEGAL_DOCS_DIR = ROOT_DIR / "data" / "raw" / "legal_docs"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess raw QA and filter to legal-doc related scope.")
    parser.add_argument("--input-file", default=str(RAW_INPUT_FILE))
    parser.add_argument("--output-file", default=str(PROCESSED_OUTPUT_FILE))
    parser.add_argument("--legal-docs-dir", default=str(LEGAL_DOCS_DIR))
    parser.add_argument("--target-count", type=int, default=500)
    parser.add_argument("--law-keyword", action="append", default=[])
    parser.add_argument(
        "--non-strict-related",
        action="store_true",
        help="Allow civil QA even when law/code hit is missing.",
    )
    parser.add_argument("--to-mongo", action="store_true")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bar output.")
    return parser.parse_args()


def load_json_list(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("items", "data", "results"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    raise SystemExit(f"Invalid list format in {path}")


def dedupe_raw_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    deduped: list[dict[str, Any]] = []
    for item in items:
        url = str(item.get("url") or "").strip()
        question = str(item.get("question") or "").strip().casefold()
        key = (url, question)
        if not url or not question or key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def progress_bar(current: int, total: int, width: int = 32) -> str:
    safe_total = max(total, 1)
    ratio = min(max(current, 0), safe_total) / safe_total
    done = int(width * ratio)
    return f"[{'#' * done}{'.' * (width - done)}] {current}/{total} ({ratio * 100:5.1f}%)"


def show_progress(prefix: str, current: int, total: int, *, final: bool = False) -> None:
    sys.stdout.write(f"\r{prefix} {progress_bar(current, total)}")
    if final:
        sys.stdout.write("\n")
    sys.stdout.flush()


def filter_items_with_progress(
    raw_items: list[dict[str, Any]],
    law_keywords: set[str],
    doc_codes: set[str],
    target_count: int,
    strict_related: bool,
    show_bar: bool,
) -> list[dict[str, Any]]:
    if not show_bar:
        return preprocess_qa_items(
            raw_items=raw_items,
            law_keywords=law_keywords,
            doc_codes=doc_codes,
            target_count=target_count,
            strict_related=strict_related,
        )

    last_pct = -1

    def callback(done: int, total: int) -> None:
        nonlocal last_pct
        if total <= 0:
            return
        pct = int(done * 100 / total)
        if pct == last_pct and done != total:
            return
        last_pct = pct
        show_progress("Filtering", done, total, final=(done == total))

    return preprocess_qa_items(
        raw_items=raw_items,
        law_keywords=law_keywords,
        doc_codes=doc_codes,
        target_count=target_count,
        strict_related=strict_related,
        progress_callback=callback,
    )


def main() -> None:
    args = parse_args()
    input_file = Path(args.input_file)
    output_file = Path(args.output_file)
    strict_related = not args.non_strict_related
    show_bar = not args.no_progress

    if not input_file.exists():
        raise SystemExit(
            f"Input file not found: {input_file}\n"
            "Run crawl first with scripts/crawl_qa_dataset.py, then preprocess."
        )

    raw_items = load_json_list(input_file)
    LOGGER.info("Loaded raw QA items (before dedupe): %s", len(raw_items))
    raw_items = dedupe_raw_items(raw_items)
    LOGGER.info("Loaded raw QA items (after dedupe): %s", len(raw_items))
    if not raw_items:
        raise SystemExit(f"No raw QA items in {input_file}")

    law_keywords, doc_codes = load_law_signals(
        legal_docs_dir=Path(args.legal_docs_dir),
        extra_keywords=args.law_keyword,
    )
    LOGGER.info("Law signals: keywords=%s doc_codes=%s", len(law_keywords), len(doc_codes))

    processed_items = filter_items_with_progress(
        raw_items=raw_items,
        law_keywords=law_keywords,
        doc_codes=doc_codes,
        target_count=args.target_count,
        strict_related=strict_related,
        show_bar=show_bar,
    )
    LOGGER.info("Current filtered QA: %s/%s", len(processed_items), args.target_count)
    if show_bar:
        show_progress("Strict target", len(processed_items), args.target_count, final=True)

    if not processed_items:
        raise SystemExit("No QA pairs after preprocessing/filtering.")

    if strict_related and len(processed_items) < args.target_count:
        LOGGER.warning(
            "Filtered only %s/%s. Increase raw data by running crawl script first.",
            len(processed_items),
            args.target_count,
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(processed_items, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Saved processed QA: %s -> %s", len(processed_items), output_file)

    if args.to_mongo:
        qa_models = [
            QAPair(
                question=item["question"],
                answer=item["answer"],
                source_url=item["url"],
                tags=item.get("tags", []),
            )
            for item in processed_items
        ]
        mongo = MongoService()
        mongo.connect()
        inserted = mongo.insert_qa_pairs(qa_models, skip_existing=True)
        skipped = max(len(qa_models) - inserted, 0)
        LOGGER.info("Inserted QA pairs into MongoDB: %s (skipped existing: %s)", inserted, skipped)


if __name__ == "__main__":
    main()
