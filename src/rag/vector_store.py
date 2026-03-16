from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from loguru import logger

from src.settings import ROOT_DIR, load_settings

try:
    import chromadb
except ImportError:  # pragma: no cover
    chromadb = None  # type: ignore[assignment]


class ChromaVectorStore:
    def __init__(self, persist_dir: str | None = None, collection_name: str | None = None) -> None:
        settings = load_settings()
        storage_cfg = settings.get("storage", {})
        resolved_dir = persist_dir or storage_cfg.get("chroma_dir", "data/vectordb/chroma")
        self.persist_dir = Path(ROOT_DIR, resolved_dir).resolve()
        self.collection_name = collection_name or storage_cfg.get("collection_name", "legal_chunks")
        self._client = None
        self._collection = None

    def connect(self) -> None:
        if chromadb is None:
            raise RuntimeError("chromadb is not installed. Run `uv sync` first.")
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self.persist_dir))
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Chroma connected: {}", self.persist_dir)

    def upsert(self, chunks: Iterable[dict[str, Any]], embeddings: list[list[float]]) -> int:
        self._ensure_connected()
        chunks_list = list(chunks)
        if not chunks_list:
            return 0
        if len(chunks_list) != len(embeddings):
            raise ValueError("Number of chunks does not match number of embeddings.")

        ids: list[str] = []
        docs: list[str] = []
        metadatas: list[dict[str, Any]] = []
        for item in chunks_list:
            ids.append(item["chunk_id"])
            docs.append(item["text"])
            metadatas.append(
                {
                    "chunk_id": item["chunk_id"],
                    "source_id": item.get("source_id", ""),
                    "title": item.get("title", ""),
                    "article": item.get("article") or "",
                    **item.get("metadata", {}),
                }
            )

        self._collection.upsert(  # type: ignore[union-attr]
            ids=ids,
            documents=docs,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        return len(ids)

    def query(
        self,
        query_embedding: list[float],
        n_results: int = 8,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        self._ensure_connected()
        result = self._collection.query(  # type: ignore[union-attr]
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        hits: list[dict[str, Any]] = []
        for idx, chunk_id in enumerate(ids):
            distance = float(distances[idx]) if idx < len(distances) else 1.0
            hits.append(
                {
                    "chunk_id": chunk_id,
                    "text": docs[idx] if idx < len(docs) else "",
                    "metadata": metas[idx] if idx < len(metas) else {},
                    "distance": distance,
                    "score": max(0.0, 1.0 - distance),
                }
            )
        return hits

    def all_chunks(self) -> list[dict[str, Any]]:
        self._ensure_connected()
        result = self._collection.get(include=["documents", "metadatas"])  # type: ignore[union-attr]
        ids = result.get("ids", [])
        docs = result.get("documents", [])
        metas = result.get("metadatas", [])
        payload: list[dict[str, Any]] = []
        for idx, chunk_id in enumerate(ids):
            payload.append(
                {
                    "chunk_id": chunk_id,
                    "text": docs[idx] if idx < len(docs) else "",
                    "metadata": metas[idx] if idx < len(metas) else {},
                }
            )
        return payload

    def count(self) -> int:
        self._ensure_connected()
        return self._collection.count()  # type: ignore[union-attr]

    def _ensure_connected(self) -> None:
        if self._collection is None:
            self.connect()

