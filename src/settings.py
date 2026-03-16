from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT_DIR / "config" / "model_config.yaml"
LOCKED_EMBEDDING_MODEL = "huyydangg/DEk21_hcmute_embedding"
DEFAULT_MODEL_CACHE_DIR = ROOT_DIR / "data" / "model_cache"


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@lru_cache(maxsize=1)
def load_settings() -> dict[str, Any]:
    load_dotenv()
    config = {}
    if CONFIG_PATH.exists():
        config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}

    llm = config.setdefault("llm", {})
    storage = config.setdefault("storage", {})
    embedding = config.setdefault("embedding", {})
    reranker = config.setdefault("reranker", {})
    retrieval = config.setdefault("retrieval", {})

    llm["host"] = os.getenv("OLLAMA_HOST", llm.get("host", "http://localhost:11434"))
    llm["model"] = os.getenv("OLLAMA_MODEL", llm.get("model", "qwen3:8b"))
    llm["timeout_seconds"] = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", llm.get("timeout_seconds", 120)))
    storage["mongo_uri"] = os.getenv("MONGODB_URI", storage.get("mongo_uri", "mongodb://localhost:27017"))
    storage["mongo_db"] = os.getenv("MONGODB_DB", storage.get("mongo_db", "vn_law_chatbot"))
    storage["chroma_dir"] = os.getenv("CHROMA_DIR", storage.get("chroma_dir", "data/vectordb/chroma"))
    embedding["model_name"] = LOCKED_EMBEDDING_MODEL
    embedding["cache_dir"] = os.getenv(
        "EMBEDDING_CACHE_DIR",
        embedding.get("cache_dir", str(DEFAULT_MODEL_CACHE_DIR)),
    )
    embedding["use_vi_tokenizer"] = _env_bool(
        "EMBEDDING_USE_VI_TOKENIZER",
        bool(embedding.get("use_vi_tokenizer", True)),
    )
    embedding.pop("fallback_models", None)
    reranker["model_name"] = os.getenv(
        "RERANKER_MODEL",
        reranker.get("model_name", "AITeamVN/Vietnamese_Reranker"),
    )
    reranker["cache_dir"] = os.getenv(
        "RERANKER_CACHE_DIR",
        reranker.get("cache_dir", embedding["cache_dir"]),
    )
    reranker["pool_size"] = int(
        os.getenv(
            "TOP_K_RERANK_POOL",
            reranker.get("pool_size", max(int(reranker.get("top_k", 4)) + 2, 6)),
        )
    )
    retrieval["top_k_final"] = int(os.getenv("TOP_K_RETRIEVAL", retrieval.get("top_k_final", 6)))
    retrieval["use_qa_collection"] = _env_bool(
        "USE_QA_COLLECTION",
        bool(retrieval.get("use_qa_collection", True)),
    )
    retrieval["qa_collection_name"] = os.getenv(
        "QA_COLLECTION_NAME",
        retrieval.get("qa_collection_name", "qa_collection"),
    )
    retrieval["top_k_dense_qa"] = int(os.getenv("TOP_K_DENSE_QA", retrieval.get("top_k_dense_qa", 4)))
    reranker["top_k"] = int(os.getenv("TOP_K_RERANK", reranker.get("top_k", 4)))

    return config
