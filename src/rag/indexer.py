from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any, Iterable

from loguru import logger

from src.processing.chunking import LegalChunker
from src.processing.preprocessor import LegalPreprocessor
from src.rag.embedder import EmbeddingService
from src.rag.vector_store import ChromaVectorStore


class LegalIndexer:
    _article_heading_pattern = re.compile(r"(?im)^\s*(?:dieu|điều)\s+(\d+[a-zA-Z0-9]*)\b")

    def __init__(
        self,
        preprocessor: LegalPreprocessor,
        chunker: LegalChunker,
        embedder: EmbeddingService,
        vector_store: ChromaVectorStore,
    ) -> None:
        self.preprocessor = preprocessor
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store

    def index_documents(
        self,
        docs: Iterable[dict[str, Any]],
        show_progress: bool = False,
        on_chunks_ready: Callable[[list[dict[str, Any]]], None] | None = None,
    ) -> int:
        docs_list = list(docs)
        total_docs = len(docs_list)
        if total_docs == 0:
            logger.warning("No input documents provided.")
            return 0

        chunks: list[dict[str, Any]] = []
        for idx, doc in enumerate(docs_list, start=1):
            source_id = str(doc.get("source_id") or doc.get("_id") or "")
            title = str(doc.get("title") or "Untitled")
            raw_text = str(doc.get("text") or "")
            if not source_id or not raw_text:
                if show_progress:
                    pct = (idx / total_docs) * 100
                    logger.info(
                        "[preprocess/chunk] {}/{} ({:.1f}%) skipped invalid doc",
                        idx,
                        total_docs,
                        pct,
                    )
                continue
            text = self.preprocessor.clean_text(raw_text)
            split_chunks = self.chunker.chunk_document(source_id=source_id, title=title, text=text)
            doc_code = str(doc.get("doc_code") or "").strip().upper()
            issued_date = str(doc.get("issued_date") or "").strip()
            effective_date = str(doc.get("effective_date") or "").strip()
            status = str(doc.get("status") or "").strip()
            source = str(doc.get("source") or "").strip()
            source_url = self._extract_source_url(doc)
            seen_chunk_signatures: set[tuple[str, str]] = set()
            for item in split_chunks:
                signature = (
                    (item.article or "").strip().lower(),
                    re.sub(r"\s+", " ", item.text or "").strip().lower(),
                )
                if signature in seen_chunk_signatures:
                    continue
                seen_chunk_signatures.add(signature)
                metadata = dict(item.metadata)
                if doc_code:
                    metadata["doc_code"] = doc_code
                if issued_date:
                    metadata["issued_date"] = issued_date
                if effective_date:
                    metadata["effective_date"] = effective_date
                if status:
                    metadata["status"] = status
                if source:
                    metadata["source"] = source
                if source_url:
                    metadata["url"] = source_url
                    metadata["source_url"] = source_url
                nested_articles = self._extract_nested_articles(item.text, item.article)
                if nested_articles:
                    metadata["nested_articles"] = nested_articles
                chunks.append(
                    {
                        "chunk_id": item.chunk_id,
                        "source_id": item.source_id,
                        "title": item.title,
                        "article": item.article,
                        "text": item.text,
                        "metadata": metadata,
                    }
                )
            if show_progress:
                pct = (idx / total_docs) * 100
                logger.info(
                    "[preprocess/chunk] {}/{} ({:.1f}%) source_id={} chunks_added={}",
                    idx,
                    total_docs,
                    pct,
                    source_id,
                    len(split_chunks),
                )

        if not chunks:
            logger.warning("No chunks produced for indexing.")
            return 0

        logger.info("Chunking done. total_docs={} total_chunks={}", total_docs, len(chunks))
        if on_chunks_ready is not None:
            on_chunks_ready(chunks)

        texts = [item["text"] for item in chunks]
        logger.info("[embedding] Start encoding {} chunks...", len(texts))
        embeddings = self.embedder.encode_texts(texts, show_progress_bar=show_progress)
        logger.info("[embedding] Done. vectors={}", len(embeddings))

        logger.info("[vectordb] Upserting chunks to Chroma...")
        inserted = self.vector_store.upsert(chunks=chunks, embeddings=embeddings)
        logger.info("[vectordb] Done. indexed_chunks={}", inserted)
        return inserted

    def _extract_source_url(self, doc: dict[str, Any]) -> str:
        raw_meta = doc.get("metadata")
        if isinstance(raw_meta, dict):
            value = raw_meta.get("url") or raw_meta.get("source_url")
            if value:
                return str(value).strip()
        direct = doc.get("url") or doc.get("source_url")
        return str(direct or "").strip()

    def _extract_nested_articles(self, text: str, article: str | None) -> str:
        hits = [m.group(1).strip() for m in self._article_heading_pattern.finditer(text or "")]
        if len(hits) < 2:
            return ""
        parent = self._article_token(article)
        nested: list[str] = []
        seen: set[str] = set()
        for token in hits:
            lowered = token.lower()
            if parent and lowered == parent:
                continue
            if lowered in seen:
                continue
            seen.add(lowered)
            nested.append(token)
            if len(nested) >= 20:
                break
        return ",".join(nested)

    def _article_token(self, article: str | None) -> str:
        if not article:
            return ""
        m = re.search(r"(\d+[a-zA-Z0-9]*)", article)
        return m.group(1).lower() if m else ""
