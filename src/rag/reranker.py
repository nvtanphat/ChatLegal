from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from loguru import logger

from src.rag.retriever import RetrievedChunk
from src.settings import load_settings

try:
    from sentence_transformers import CrossEncoder
except ImportError:  # pragma: no cover
    CrossEncoder = None  # type: ignore[assignment]


class VietnameseReranker:
    def __init__(self, enabled: bool | None = None, model_name: str | None = None) -> None:
        cfg = load_settings().get("reranker", {})
        self.enabled = bool(cfg.get("enabled", True)) if enabled is None else enabled
        self.model_name = model_name or cfg.get("model_name", "AITeamVN/Vietnamese_Reranker")
        self.top_k = int(cfg.get("top_k", 4))
        self.pool_size = int(cfg.get("pool_size", max(self.top_k + 2, 6)))
        self.cache_dir = str(cfg.get("cache_dir", "")).strip() or None
        self._model = None

    def rerank(
        self,
        query: str,
        candidates: Iterable[RetrievedChunk],
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        items = list(candidates)
        if not items:
            return []

        limit = top_k or self.top_k
        if not self.enabled:
            return sorted(items, key=lambda x: x.final_score, reverse=True)[:limit]

        try:
            ranked_by_hybrid = sorted(items, key=lambda x: x.final_score, reverse=True)
            pool_limit = max(limit, self.pool_size)
            rerank_pool = ranked_by_hybrid[:pool_limit]

            self._ensure_model()
            pairs = [(query, item.text) for item in rerank_pool]
            scores = self._model.predict(pairs)  # type: ignore[union-attr]
            for item, score in zip(rerank_pool, scores):
                item.rerank_score = float(score)
            rerank_pool.sort(key=lambda x: x.rerank_score or -1.0, reverse=True)
            return rerank_pool[:limit]
        except Exception as exc:
            logger.warning("Reranker failed, fallback to hybrid score: {}", exc)
            return sorted(items, key=lambda x: x.final_score, reverse=True)[:limit]

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        if CrossEncoder is None:
            raise RuntimeError("sentence-transformers is not installed. Run `uv sync` first.")
        logger.info("Loading reranker model: {}", self.model_name)
        cache_folder = self._prepare_cache_dir()
        self._model = CrossEncoder(self.model_name, cache_folder=cache_folder)

    def _prepare_cache_dir(self) -> str | None:
        if not self.cache_dir:
            return None
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        return self.cache_dir
