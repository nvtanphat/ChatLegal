from __future__ import annotations

from typing import Iterable

from src.prompts.templates import LEGAL_SYSTEM_PROMPT, LEGAL_USER_TEMPLATE
from src.rag.retriever import RetrievedChunk


class PromptChain:
    def __init__(self, max_context_chars: int = 6000) -> None:
        self.max_context_chars = max_context_chars
        self.system_prompt = LEGAL_SYSTEM_PROMPT

    def build_context(self, chunks: Iterable[RetrievedChunk]) -> str:
        lines: list[str] = []
        current_size = 0
        for idx, item in enumerate(chunks, start=1):
            article = item.article or "N/A"
            metadata = item.metadata or {}
            source_kind = metadata.get("source_kind", "legal")
            score = item.rerank_score if item.rerank_score is not None else item.final_score
            block = (
                f"[{idx}] source_kind={source_kind} | source_id={item.source_id} | "
                f"article={article} | title={item.title} | score={score:.3f}\n"
                f"{item.text.strip()}\n"
            )
            if current_size + len(block) > self.max_context_chars:
                break
            lines.append(block)
            current_size += len(block)
        return "\n".join(lines).strip()

    def build_user_prompt(self, query: str, chunks: Iterable[RetrievedChunk]) -> str:
        context = self.build_context(chunks)
        return LEGAL_USER_TEMPLATE.format(query=query.strip(), context=context or "(khong co du lieu)")
