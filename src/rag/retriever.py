from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from src.processing.tokenizer import tokenize_vi
from src.rag.embedder import EmbeddingService
from src.rag.vector_store import ChromaVectorStore
from src.settings import load_settings


@dataclass(slots=True)
class RetrievedChunk:
    chunk_id: str
    source_id: str
    title: str
    article: str | None
    text: str
    dense_score: float
    lexical_score: float
    final_score: float
    rerank_score: float | None = None
    metadata: dict[str, Any] | None = None


class BM25Lite:
    def __init__(self, corpus_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.corpus_tokens = corpus_tokens
        self.k1 = k1
        self.b = b
        self.doc_count = len(corpus_tokens)
        self.avgdl = (
            sum(len(tokens) for tokens in corpus_tokens) / self.doc_count if self.doc_count else 0.0
        )
        self.doc_freq: dict[str, int] = defaultdict(int)
        self.term_freqs: list[Counter[str]] = []
        for tokens in corpus_tokens:
            counts = Counter(tokens)
            self.term_freqs.append(counts)
            for token in counts:
                self.doc_freq[token] += 1

    def score_query(self, query_tokens: list[str], doc_index: int) -> float:
        if self.doc_count == 0:
            return 0.0
        score = 0.0
        doc_tf = self.term_freqs[doc_index]
        dl = len(self.corpus_tokens[doc_index]) or 1
        for token in query_tokens:
            df = self.doc_freq.get(token, 0)
            if df == 0:
                continue
            idf = math.log((self.doc_count - df + 0.5) / (df + 0.5) + 1)
            tf = doc_tf.get(token, 0)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
            score += idf * (numerator / (denominator or 1))
        return score

    def top_k(self, query_tokens: list[str], k: int) -> list[tuple[int, float]]:
        scores = [(idx, self.score_query(query_tokens, idx)) for idx in range(self.doc_count)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return [item for item in scores[:k] if item[1] > 0]


class HybridRetriever:
    def __init__(
        self,
        vector_store: ChromaVectorStore,
        embedder: EmbeddingService,
        lexical_chunks: list[dict[str, Any]] | None = None,
        qa_vector_store: ChromaVectorStore | None = None,
    ) -> None:
        settings = load_settings().get("retrieval", {})
        multi_query_cfg = settings.get("multi_query", {}) or {}
        hyde_cfg = settings.get("hyde", {}) or {}
        metadata_filter_cfg = settings.get("metadata_filter", {}) or {}

        self.top_k_dense = int(settings.get("top_k_dense", 8))
        self.top_k_dense_qa = int(settings.get("top_k_dense_qa", 4))
        self.top_k_lexical = int(settings.get("top_k_lexical", 8))
        self.top_k_final = int(settings.get("top_k_final", 6))
        self.min_score = float(settings.get("min_score", 0.0))
        self.multi_query_dense_weight = float(multi_query_cfg.get("dense_weight", 0.9))
        self.multi_query_lexical_weight = float(multi_query_cfg.get("lexical_weight", 0.9))
        self.hyde_dense_weight = float(hyde_cfg.get("dense_weight", 0.8))
        self.filter_fallback_unfiltered = bool(metadata_filter_cfg.get("fallback_unfiltered", True))
        self.use_qa_collection = bool(settings.get("use_qa_collection", True))
        self.qa_collection_name = str(settings.get("qa_collection_name", "qa_collection")).strip() or "qa_collection"
        self.vector_store = vector_store
        self.vector_store_collection_name = str(
            getattr(self.vector_store, "collection_name", settings.get("collection_name", "legal_chunks"))
        ).strip() or "legal_chunks"
        self.vector_store_persist_dir = getattr(self.vector_store, "persist_dir", None)
        self.qa_vector_store: ChromaVectorStore | None = None
        if self.use_qa_collection:
            if qa_vector_store is not None:
                self.qa_vector_store = qa_vector_store
            elif (
                self.qa_collection_name != self.vector_store_collection_name
                and self.vector_store_persist_dir is not None
            ):
                self.qa_vector_store = ChromaVectorStore(
                    persist_dir=str(self.vector_store_persist_dir),
                    collection_name=self.qa_collection_name,
                )
        self.embedder = embedder
        self.lexical_chunks = lexical_chunks or []
        self._bm25: BM25Lite | None = None
        if self.lexical_chunks:
            self.build_lexical_index(self.lexical_chunks)

    def build_lexical_index(self, chunks: list[dict[str, Any]]) -> None:
        self.lexical_chunks = chunks
        corpus = [tokenize_vi(item.get("text", "")) for item in chunks]
        self._bm25 = BM25Lite(corpus_tokens=corpus)

    def search(
        self,
        query: str,
        extra_queries: list[str] | None = None,
        hyde_answer: str | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        outputs = self._search_internal(
            query=query,
            extra_queries=extra_queries,
            hyde_answer=hyde_answer,
            metadata_filter=metadata_filter,
        )
        if outputs or metadata_filter is None or not self.filter_fallback_unfiltered:
            return outputs
        return self._search_internal(
            query=query,
            extra_queries=extra_queries,
            hyde_answer=hyde_answer,
            metadata_filter=None,
        )

    def _search_internal(
        self,
        query: str,
        extra_queries: list[str] | None = None,
        hyde_answer: str | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        dense_hits: list[dict[str, Any]] = []
        lexical_hits: list[dict[str, Any]] = []

        normalized_query = (query or "").strip()
        expanded_queries = self._dedupe_queries(extra_queries)

        dense_queries: list[tuple[str, float]] = [(normalized_query, 1.0)]
        for item in expanded_queries:
            dense_queries.append((item, self.multi_query_dense_weight))
        if (hyde_answer or "").strip():
            dense_queries.append((hyde_answer.strip(), self.hyde_dense_weight))

        for dense_query, weight in dense_queries:
            if not dense_query:
                continue
            query_embedding = self.embedder.encode_query(dense_query)
            dense_hits.extend(
                self._dense_search(
                    query_embedding=query_embedding,
                    metadata_filter=metadata_filter,
                    score_multiplier=weight,
                )
            )

        lexical_queries: list[tuple[str, float]] = [(normalized_query, 1.0)]
        for item in expanded_queries:
            lexical_queries.append((item, self.multi_query_lexical_weight))

        for lexical_query, weight in lexical_queries:
            if not lexical_query:
                continue
            lexical_hits.extend(
                self._lexical_search(
                    lexical_query,
                    metadata_filter=metadata_filter,
                    score_multiplier=weight,
                )
            )

        dense_hits = self._aggregate_hits(dense_hits)
        lexical_hits = self._aggregate_hits(lexical_hits)
        return self._fuse(query=normalized_query, dense_hits=dense_hits, lexical_hits=lexical_hits)

    def _dedupe_queries(self, queries: list[str] | None) -> list[str]:
        if not queries:
            return []
        outputs: list[str] = []
        seen: set[str] = set()
        for item in queries:
            text = str(item or "").strip()
            if not text:
                continue
            key = text.casefold()
            if key in seen:
                continue
            seen.add(key)
            outputs.append(text)
        return outputs

    def _dense_search(
        self,
        query_embedding: list[float],
        metadata_filter: dict[str, Any] | None = None,
        score_multiplier: float = 1.0,
    ) -> list[dict[str, Any]]:
        hits: list[dict[str, Any]] = []
        hits.extend(
            self._query_store(
                store=self.vector_store,
                query_embedding=query_embedding,
                n_results=self.top_k_dense,
                where=metadata_filter,
                score_multiplier=score_multiplier,
            )
        )
        if self.qa_vector_store is not None and self.top_k_dense_qa > 0:
            hits.extend(
                self._query_store(
                    store=self.qa_vector_store,
                    query_embedding=query_embedding,
                    n_results=self.top_k_dense_qa,
                    where=metadata_filter,
                    score_multiplier=score_multiplier,
                )
            )
        if metadata_filter is not None:
            hits = self._filter_hits_by_metadata(hits, metadata_filter)
        return hits

    def _query_store(
        self,
        store: ChromaVectorStore,
        query_embedding: list[float],
        n_results: int,
        where: dict[str, Any] | None = None,
        score_multiplier: float = 1.0,
    ) -> list[dict[str, Any]]:
        collection_name = str(getattr(store, "collection_name", self.vector_store_collection_name))
        query_kwargs: dict[str, Any] = {
            "query_embedding": query_embedding,
            "n_results": n_results,
        }
        if where is not None:
            query_kwargs["where"] = where

        try:
            raw_hits = store.query(**query_kwargs)
        except TypeError:
            raw_hits = store.query(query_embedding=query_embedding, n_results=n_results)
        except Exception:
            if where is None:
                raise
            raw_hits = store.query(query_embedding=query_embedding, n_results=n_results)

        outputs: list[dict[str, Any]] = []
        for hit in raw_hits:
            shaped = self._shape_hit(hit=hit, collection_name=collection_name)
            shaped["score"] = float(shaped.get("score", 0.0)) * max(score_multiplier, 0.0)
            outputs.append(shaped)
        return outputs

    def _shape_hit(self, hit: dict[str, Any], collection_name: str) -> dict[str, Any]:
        metadata = dict(hit.get("metadata", {}) or {})
        chunk_id = str(hit.get("chunk_id", ""))
        text = str(hit.get("text", "") or "")

        metadata["collection"] = collection_name
        if collection_name == self.qa_collection_name:
            metadata["source_kind"] = "qa"
            question = str(metadata.get("question", "") or "").strip()
            answer = str(metadata.get("answer", "") or "").strip()
            source_url = str(metadata.get("source_url", "") or "").strip()
            if answer:
                if question:
                    text = f"Cau hoi tham khao: {question}\nTra loi tham khao: {answer}"
                else:
                    text = f"Tra loi tham khao: {answer}"
            metadata["source_id"] = source_url or str(metadata.get("source_id", "") or "") or "qa_pairs"
            metadata["title"] = str(metadata.get("title", "") or "").strip() or "QA Pair"
            metadata["article"] = ""
        else:
            metadata["source_kind"] = "legal"

        return {
            "retrieval_key": f"{collection_name}:{chunk_id}",
            "chunk_id": chunk_id,
            "text": text,
            "metadata": metadata,
            "score": float(hit.get("score", 0.0)),
        }

    def _lexical_search(
        self,
        query: str,
        metadata_filter: dict[str, Any] | None = None,
        score_multiplier: float = 1.0,
    ) -> list[dict[str, Any]]:
        if self._bm25 is None and not self.lexical_chunks:
            if not hasattr(self.vector_store, "all_chunks"):
                return []
            self.build_lexical_index(self.vector_store.all_chunks())
        if self._bm25 is None:
            return []
        tokens = tokenize_vi(query)
        top = self._bm25.top_k(tokens, self.top_k_lexical)
        hits: list[dict[str, Any]] = []
        for idx, score in top:
            item = self.lexical_chunks[idx]
            meta = dict(item.get("metadata", {}) or {})
            meta.setdefault("collection", self.vector_store_collection_name)
            meta.setdefault("source_kind", "legal")
            if metadata_filter is not None and not self._metadata_match(meta, metadata_filter):
                continue
            chunk_id = str(item.get("chunk_id", ""))
            hits.append(
                {
                    "retrieval_key": f"{self.vector_store_collection_name}:{chunk_id}",
                    "chunk_id": chunk_id,
                    "text": item.get("text", ""),
                    "metadata": meta,
                    "score": float(score) * max(score_multiplier, 0.0),
                }
            )
        return hits

    def _aggregate_hits(self, hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}
        for hit in hits:
            retrieval_key = str(hit.get("retrieval_key") or hit.get("chunk_id", ""))
            if not retrieval_key:
                continue
            current = merged.get(retrieval_key)
            if current is None:
                merged[retrieval_key] = {
                    "retrieval_key": retrieval_key,
                    "chunk_id": str(hit.get("chunk_id", "")),
                    "text": hit.get("text", ""),
                    "metadata": hit.get("metadata", {}) or {},
                    "score": float(hit.get("score", 0.0)),
                }
                continue
            current["score"] = float(current.get("score", 0.0)) + float(hit.get("score", 0.0))
        return list(merged.values())

    def _filter_hits_by_metadata(
        self,
        hits: list[dict[str, Any]],
        metadata_filter: dict[str, Any],
    ) -> list[dict[str, Any]]:
        return [
            hit
            for hit in hits
            if self._metadata_match(hit.get("metadata", {}) or {}, metadata_filter)
        ]

    def _metadata_match(self, metadata: dict[str, Any], where: dict[str, Any]) -> bool:
        if not where:
            return True
        if "$and" in where:
            conditions = where.get("$and") or []
            if not isinstance(conditions, list):
                return True
            return all(self._metadata_match(metadata, cond) for cond in conditions if isinstance(cond, dict))
        if "$or" in where:
            conditions = where.get("$or") or []
            if not isinstance(conditions, list):
                return True
            return any(self._metadata_match(metadata, cond) for cond in conditions if isinstance(cond, dict))

        for key, condition in where.items():
            if key.startswith("$"):
                continue
            value = metadata.get(key)
            if isinstance(condition, dict):
                if "$eq" in condition and str(value or "") != str(condition.get("$eq") or ""):
                    return False
                if "$ne" in condition and str(value or "") == str(condition.get("$ne") or ""):
                    return False
                continue
            if str(value or "") != str(condition or ""):
                return False
        return True

    def _fuse(
        self,
        query: str,
        dense_hits: list[dict[str, Any]],
        lexical_hits: list[dict[str, Any]],
    ) -> list[RetrievedChunk]:
        dense_norm = self._normalize_scores(dense_hits)
        lexical_norm = self._normalize_scores(lexical_hits)

        merged: dict[str, dict[str, Any]] = {}
        for hit, score in zip(dense_hits, dense_norm):
            merge_key = str(hit.get("retrieval_key") or hit.get("chunk_id", ""))
            merged[merge_key] = {
                "chunk_id": str(hit.get("chunk_id", "")),
                "text": hit.get("text", ""),
                "metadata": hit.get("metadata", {}) or {},
                "dense_score": score,
                "lexical_score": 0.0,
            }

        for hit, score in zip(lexical_hits, lexical_norm):
            merge_key = str(hit.get("retrieval_key") or hit.get("chunk_id", ""))
            if merge_key not in merged:
                merged[merge_key] = {
                    "chunk_id": str(hit.get("chunk_id", "")),
                    "text": hit.get("text", ""),
                    "metadata": hit.get("metadata", {}) or {},
                    "dense_score": 0.0,
                    "lexical_score": score,
                }
            else:
                merged[merge_key]["lexical_score"] = score

        outputs: list[RetrievedChunk] = []
        for _, item in merged.items():
            dense_score = float(item["dense_score"])
            lexical_score = float(item["lexical_score"])
            final_score = 0.65 * dense_score + 0.35 * lexical_score
            if final_score < self.min_score:
                continue
            metadata = item["metadata"]
            outputs.append(
                RetrievedChunk(
                    chunk_id=item["chunk_id"],
                    source_id=metadata.get("source_id", ""),
                    title=metadata.get("title", ""),
                    article=metadata.get("article", "") or None,
                    text=item["text"],
                    dense_score=dense_score,
                    lexical_score=lexical_score,
                    final_score=final_score,
                    metadata=metadata,
                )
            )

        outputs.sort(key=lambda x: x.final_score, reverse=True)
        return outputs[: self.top_k_final]

    @staticmethod
    def _normalize_scores(items: list[dict[str, Any]]) -> list[float]:
        if not items:
            return []
        raw_scores = [float(item.get("score", 0.0)) for item in items]
        low = min(raw_scores)
        high = max(raw_scores)
        if math.isclose(high, low):
            return [1.0 if high > 0 else 0.0 for _ in raw_scores]
        return [(score - low) / (high - low) for score in raw_scores]
