from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from loguru import logger

from src.database.mongo_client import MongoService
from src.processing.chunking import LegalChunker
from src.processing.preprocessor import LegalPreprocessor
from src.rag.embedder import EmbeddingService
from src.rag.indexer import LegalIndexer
from src.rag.vector_store import ChromaVectorStore
from src.settings import ROOT_DIR

DEFAULT_PROCESSED_OUTPUT_FILE = ROOT_DIR / "data" / "processed" / "legal_docs" / "legal_chunks.json"


def load_docs_from_json_dir(json_dir: Path) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    if not json_dir.exists():
        return docs
    for file_path in sorted(json_dir.glob("*.json")):
        try:
            docs.append(json.loads(file_path.read_text(encoding="utf-8")))
        except Exception as exc:
            logger.warning("Skip invalid file {}: {}", file_path, exc)
    return docs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build embeddings and index legal docs")
    parser.add_argument("--from-mongo", action="store_true", help="Load documents from MongoDB")
    parser.add_argument(
        "--json-dir",
        default=str(ROOT_DIR / "data" / "raw" / "legal_docs"),
        help="Directory containing crawled legal doc JSON files",
    )
    parser.add_argument("--chunk-size", type=int, default=1600)
    parser.add_argument("--chunk-overlap", type=int, default=120)
    parser.add_argument(
        "--processed-output-file",
        default=str(DEFAULT_PROCESSED_OUTPUT_FILE),
        help="Path to save preprocessed/chunked legal data as JSON",
    )
    parser.add_argument(
        "--no-save-processed",
        action="store_true",
        help="Disable saving preprocessed/chunked legal data to file.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress output",
    )
    return parser.parse_args()


def save_processed_chunks(chunks: list[dict[str, Any]], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Saved processed legal chunks: {} -> {}", len(chunks), output_file)


def main() -> None:
    args = parse_args()
    show_progress = not args.no_progress

    docs: list[dict[str, Any]]
    if args.from_mongo:
        logger.info("Loading documents from MongoDB...")
        mongo = MongoService()
        mongo.connect()
        docs = list(mongo.stream_legal_docs())
        logger.info("Loaded {} docs from MongoDB", len(docs))
    else:
        logger.info("Loading documents from JSON directory...")
        docs = load_docs_from_json_dir(Path(args.json_dir))
        logger.info("Loaded {} docs from {}", len(docs), args.json_dir)

    if not docs:
        raise SystemExit("No legal documents found to index.")

    preprocessor = LegalPreprocessor()
    chunker = LegalChunker(max_chars=args.chunk_size, overlap_chars=args.chunk_overlap)
    embedder = EmbeddingService()
    vector_store = ChromaVectorStore()
    indexer = LegalIndexer(
        preprocessor=preprocessor,
        chunker=chunker,
        embedder=embedder,
        vector_store=vector_store,
    )
    chunks_callback = None
    if not args.no_save_processed:
        output_file = Path(args.processed_output_file)

        def _chunks_callback(chunks: list[dict[str, Any]]) -> None:
            save_processed_chunks(chunks, output_file)

        chunks_callback = _chunks_callback

    indexed = indexer.index_documents(
        docs,
        show_progress=show_progress,
        on_chunks_ready=chunks_callback,
    )
    logger.info("Completed indexing. Total chunks: {}", indexed)


if __name__ == "__main__":
    main()
