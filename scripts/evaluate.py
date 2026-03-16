from __future__ import annotations

import argparse
import json
from pathlib import Path

from loguru import logger

from src.core.model_factory import create_llm
from src.evaluation.golden_dataset import load_golden_dataset
from src.evaluation.ragas_eval import run_basic_eval
from src.inference.inference_engine import InferenceEngine
from src.rag.embedder import EmbeddingService
from src.rag.reranker import VietnameseReranker
from src.rag.retriever import HybridRetriever
from src.rag.vector_store import ChromaVectorStore
from src.settings import ROOT_DIR

DEFAULT_FALLBACK_QA_PATH = ROOT_DIR / "data" / "processed" / "qa_pairs" / "qa_pairs_processed_vbpl_civil_500.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run basic evaluation for VN Law Chatbot.")
    parser.add_argument(
        "--dataset-file",
        default="",
        help="Golden dataset path (default: data/qa_dataset/golden_dataset.json).",
    )
    parser.add_argument(
        "--fallback-qa-file",
        default=str(DEFAULT_FALLBACK_QA_PATH),
        help="Fallback QA file used when golden dataset is missing/empty.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=30,
        help="Maximum number of questions to evaluate (0 = all).",
    )
    return parser.parse_args()


def _trim_samples(samples: list[dict], max_samples: int) -> list[dict]:
    if max_samples <= 0:
        return samples
    return samples[:max_samples]


def load_eval_samples(dataset_file: str, fallback_qa_file: str, max_samples: int) -> list[dict]:
    golden = load_golden_dataset(dataset_file or None)
    if golden:
        logger.info("Using golden dataset: {} (samples={})", dataset_file or "default", len(golden))
        return _trim_samples(golden, max_samples)

    fallback_path = Path(fallback_qa_file)
    if not fallback_path.exists():
        logger.warning("No golden dataset and fallback QA file not found: {}", fallback_path)
        return []
    rows = json.loads(fallback_path.read_text(encoding="utf-8"))
    samples = [{"question": str(row.get("question") or "")} for row in rows if str(row.get("question") or "").strip()]
    logger.info("Using fallback QA dataset: {} (samples={})", fallback_path, len(samples))
    return _trim_samples(samples, max_samples)


def main() -> None:
    args = parse_args()
    llm = create_llm()
    embedder = EmbeddingService()
    vector_store = ChromaVectorStore()
    retriever = HybridRetriever(vector_store=vector_store, embedder=embedder)
    reranker = VietnameseReranker()
    engine = InferenceEngine(llm=llm, retriever=retriever, reranker=reranker, mongo_service=None)

    eval_samples = load_eval_samples(
        dataset_file=args.dataset_file,
        fallback_qa_file=args.fallback_qa_file,
        max_samples=args.max_samples,
    )
    result = run_basic_eval(engine=engine, golden_samples=eval_samples)
    logger.info("Eval total={} answered={} answer_rate={:.2%}", result.total, result.answered, result.answer_rate)


if __name__ == "__main__":
    main()
