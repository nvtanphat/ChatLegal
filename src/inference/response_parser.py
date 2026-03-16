from __future__ import annotations

import math
from typing import Iterable

from src.rag.retriever import RetrievedChunk


class ResponseParser:
    def clean_answer(self, text: str) -> str:
        text = (text or "").strip()
        if not text:
            return "Xin loi, toi chua tao duoc cau tra loi."
        return text

    def enforce_legal_opening(self, answer: str, chunks: Iterable[RetrievedChunk]) -> str:
        text = (answer or "").strip()
        if not text:
            return text

        lowered = text.lower()
        if lowered.startswith("theo "):
            return text

        items = list(chunks)
        if not items:
            return text

        legal_first = next(
            (
                item
                for item in items
                if str((item.metadata or {}).get("source_kind", "")).strip().lower() == "legal"
            ),
            items[0],
        )

        source_id = str(legal_first.source_id or "").strip()
        article = str(legal_first.article or "").strip()
        title = str(legal_first.title or "").strip()

        if source_id and article:
            opener = f"Theo {source_id}, {article}: "
        elif title and article:
            opener = f"Theo {title}, {article}: "
        elif article:
            opener = f"Theo {article}: "
        elif source_id:
            opener = f"Theo {source_id}: "
        else:
            opener = "Theo quy dinh phap luat hien hanh: "

        return f"{opener}{text}"

    def make_citations(self, chunks: Iterable[RetrievedChunk]) -> list[dict]:
        citations: list[dict] = []
        for item in chunks:
            score = float(item.final_score)
            if not math.isfinite(score):
                score = 0.0
            citations.append(
                {
                    "chunk_id": item.chunk_id,
                    "source_id": item.source_id,
                    "title": item.title,
                    "article": item.article,
                    "text": item.text,
                    "score": round(score, 4),
                    "source_kind": str((item.metadata or {}).get("source_kind", "")).strip().lower(),
                }
            )
        return citations
