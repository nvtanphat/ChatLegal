from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

from loguru import logger

from src.settings import ROOT_DIR, load_settings

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover
    SentenceTransformer = None  # type: ignore[assignment]

try:
    from pyvi import ViTokenizer
except ImportError:  # pragma: no cover
    ViTokenizer = None  # type: ignore[assignment]


class EmbeddingService:
    LOCKED_MODEL_NAME = "huyydangg/DEk21_hcmute_embedding"

    def __init__(
        self,
        model_name: str | None = None,
        batch_size: int | None = None,
        normalize_embeddings: bool | None = None,
        fallback_models: list[str] | None = None,
        use_vi_tokenizer: bool | None = None,
    ) -> None:
        settings = load_settings()
        emb_cfg = settings.get("embedding", {})
        requested_model = model_name or emb_cfg.get("model_name")
        if requested_model and requested_model != self.LOCKED_MODEL_NAME:
            logger.warning(
                "Embedding model '{}' is ignored. System is locked to '{}'.",
                requested_model,
                self.LOCKED_MODEL_NAME,
            )
        if fallback_models:
            logger.warning("fallback_models is ignored. Only '{}' is allowed.", self.LOCKED_MODEL_NAME)

        self.model_name = self.LOCKED_MODEL_NAME
        self.batch_size = batch_size or int(emb_cfg.get("batch_size", 32))
        self.normalize_embeddings = (
            normalize_embeddings
            if normalize_embeddings is not None
            else bool(emb_cfg.get("normalize_embeddings", True))
        )
        self.use_vi_tokenizer = (
            bool(emb_cfg.get("use_vi_tokenizer", True))
            if use_vi_tokenizer is None
            else use_vi_tokenizer
        )
        configured_cache = str(emb_cfg.get("cache_dir", "")).strip()
        default_cache = ROOT_DIR / "data" / "model_cache"
        if configured_cache:
            self.cache_dir = configured_cache
        elif default_cache.exists():
            self.cache_dir = str(default_cache)
        else:
            self.cache_dir = None
        self._model = None
        self.loaded_model_name: str | None = None
        self._warned_missing_pyvi = False

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers is not installed. Run `uv sync` first.")
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
        cache_folder = self._prepare_cache_dir()
        load_kwargs: dict[str, str] = {}
        if token:
            load_kwargs["token"] = token
        if cache_folder:
            load_kwargs["cache_folder"] = cache_folder

        local_only_error: Exception | None = None
        try:
            logger.info("Loading embedding model: {}", self.model_name)
            self._model = SentenceTransformer(self.model_name, local_files_only=True, **load_kwargs)
            self.loaded_model_name = self.model_name
            return
        except Exception as exc:  # pragma: no cover
            local_only_error = exc
            logger.warning(
                "Local-only load failed for '{}': {}. Retrying with network...",
                self.model_name,
                exc,
            )
        try:
            self._model = SentenceTransformer(self.model_name, **load_kwargs)
            self.loaded_model_name = self.model_name
            return
        except Exception as exc:  # pragma: no cover
            details = f"{exc}"
            if local_only_error is not None:
                details = f"local_only={local_only_error}; online={exc}"
            raise RuntimeError(
                "Cannot load locked embedding model "
                f"'{self.model_name}'. Please check Hugging Face access (`hf auth login`) and network. "
                f"Details: {details}"
            ) from exc

    def encode_texts(self, texts: Sequence[str], show_progress_bar: bool = False) -> list[list[float]]:
        self._ensure_model()
        clean_texts = [text.strip() for text in texts if text and text.strip()]
        if not clean_texts:
            return []
        clean_texts = self._prepare_for_embedding(clean_texts)
        vectors = self._model.encode(  # type: ignore[union-attr]
            clean_texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=show_progress_bar,
        )
        return vectors.tolist()

    def encode_query(self, query: str) -> list[float]:
        vectors = self.encode_texts([query], show_progress_bar=False)
        if not vectors:
            raise ValueError("Query is empty.")
        return vectors[0]

    def _prepare_for_embedding(self, texts: Sequence[str]) -> list[str]:
        if not self._should_apply_vi_tokenizer():
            return list(texts)

        if ViTokenizer is None:
            if not self._warned_missing_pyvi:
                logger.warning(
                    "EMBEDDING_USE_VI_TOKENIZER=true but pyvi is not installed. Continue without segmentation."
                )
                self._warned_missing_pyvi = True
            return list(texts)

        return [ViTokenizer.tokenize(text) for text in texts]

    def _should_apply_vi_tokenizer(self) -> bool:
        if not self.use_vi_tokenizer:
            return False
        return True

    def _prepare_cache_dir(self) -> str | None:
        if not self.cache_dir:
            return None
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        return self.cache_dir
