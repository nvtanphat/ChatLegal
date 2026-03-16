from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from loguru import logger

from src.rag.embedder import EmbeddingService
from src.rag.vector_store import ChromaVectorStore
from src.settings import ROOT_DIR

DEFAULT_INPUT_FILE = (
    ROOT_DIR / "data" / "processed" / "qa_pairs" / "qa_pairs_processed_vbpl_civil_500.json"
)
DEFAULT_COLLECTION_NAME = "qa_collection"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build embeddings for QA dataset and index into a dedicated Chroma collection."
    )
    parser.add_argument("--input-file", default=str(DEFAULT_INPUT_FILE))
    parser.add_argument("--collection-name", default=DEFAULT_COLLECTION_NAME)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of QA pairs to index (0 = all).",
    )
    parser.add_argument("--no-progress", action="store_true", help="Disable embedding progress bar.")
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def load_json_list(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"Input file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise SystemExit(f"Invalid list format in {path}")
    return [item for item in payload if isinstance(item, dict)]


def to_chroma_metadata_value(value: Any) -> str | int | float | bool:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        compact = [normalize_text(item) for item in value]
        compact = [item for item in compact if item]
        return " | ".join(compact)
    return json.dumps(value, ensure_ascii=False)


def sanitize_metadata(raw: dict[str, Any]) -> dict[str, str | int | float | bool]:
    return {key: to_chroma_metadata_value(value) for key, value in raw.items()}


def build_question_id(url: str, question: str, index: int) -> str:
    digest = hashlib.sha1(f"{url}|{question}".encode("utf-8")).hexdigest()
    return f"qa_{digest}_{index}"


def prepare_qa_chunks(items: list[dict[str, Any]], limit: int = 0) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()

    for idx, item in enumerate(items, start=1):
        if limit > 0 and len(chunks) >= limit:
            break

        question = normalize_text(item.get("question"))
        answer = normalize_text(item.get("answer"))
        source_url = normalize_text(item.get("url") or item.get("source_url"))
        topic = normalize_text(item.get("topic") or "qa")
        if len(question) < 12 or len(answer) < 20:
            continue

        pair_key = (source_url.casefold(), question.casefold())
        if pair_key in seen_pairs:
            continue

        metadata = sanitize_metadata(
            {
                "question": question,
                "answer": answer,
                "source_url": source_url,
                "source": normalize_text(item.get("source")),
                "source_site": normalize_text(item.get("source_site")),
                "scope": normalize_text(item.get("scope")),
                "topic": topic,
                "tags": item.get("tags", []),
                "cited_laws": item.get("cited_laws", []),
                "cited_articles": item.get("cited_articles", []),
                "crawled_at": normalize_text(item.get("crawled_at")),
            }
        )

        chunks.append(
            {
                "chunk_id": build_question_id(source_url, question, idx),
                "source_id": source_url or "qa_pairs",
                "title": topic,
                "article": "",
                "text": question,
                "metadata": metadata,
            }
        )
        seen_pairs.add(pair_key)

    return chunks


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_file)
    show_progress = not args.no_progress

    qa_items = load_json_list(input_path)
    logger.info("Loaded {} QA items from {}", len(qa_items), input_path)

    qa_chunks = prepare_qa_chunks(qa_items, limit=max(args.limit, 0))
    logger.info("Prepared {} QA records for indexing", len(qa_chunks))
    if not qa_chunks:
        raise SystemExit("No QA records after preprocessing.")

    questions = [item["text"] for item in qa_chunks]
    embedder = EmbeddingService()
    logger.info("[embedding] Start encoding {} QA questions...", len(questions))
    embeddings = embedder.encode_texts(questions, show_progress_bar=show_progress)
    logger.info("[embedding] Done. vectors={}", len(embeddings))

    vector_store = ChromaVectorStore(collection_name=args.collection_name)
    logger.info("[vectordb] Upserting QA vectors to collection '{}'...", args.collection_name)
    inserted = vector_store.upsert(chunks=qa_chunks, embeddings=embeddings)
    total = vector_store.count()
    logger.info("[vectordb] Done. upserted={} total_in_collection={}", inserted, total)


if __name__ == "__main__":
    main()
