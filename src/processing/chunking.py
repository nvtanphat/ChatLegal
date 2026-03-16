from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    source_id: str
    title: str
    article: str | None
    text: str
    metadata: dict[str, Any]


class LegalChunker:
    # Matches forms like "Dieu 1.", "Điều 124", "DIEU 5:", "Dieu 10a"
    article_pattern = re.compile(r"(?im)^\s*(?:dieu|điều)\s+(\d+[a-zA-Z0-9]*)[\.:]?\s")
    article_line_pattern = re.compile(r"(?im)^\s*(?:dieu|điều)\s+\d+[a-zA-Z0-9]*[\.:]?\s")

    max_chars: int
    overlap_chars: int

    def __init__(self, max_chars: int = 1600, overlap_chars: int = 120) -> None:
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars

    def chunk_document(self, source_id: str, title: str, text: str) -> list[Chunk]:
        text = (text or "").strip()
        if not text:
            return []

        article_chunks = self._split_by_articles(source_id=source_id, title=title, text=text)
        if article_chunks:
            final_chunks: list[Chunk] = []
            for chunk in article_chunks:
                if len(chunk.text) <= self.max_chars:
                    final_chunks.append(chunk)
                else:
                    final_chunks.extend(self._split_long_chunk(chunk))
            return final_chunks

        return self._window_chunk(source_id=source_id, title=title, text=text)

    def _split_by_articles(self, source_id: str, title: str, text: str) -> list[Chunk]:
        lines = text.splitlines()
        chunks: list[Chunk] = []
        current_article: str | None = None
        current_lines: list[str] = []
        seen_signatures: set[tuple[str, str]] = set()

        def flush() -> None:
            if not current_lines:
                return
            content = "\n".join(current_lines).strip()
            if not content:
                return
            signature = (
                (current_article or "").strip().lower(),
                re.sub(r"\s+", " ", content).strip().lower(),
            )
            if signature in seen_signatures:
                return
            seen_signatures.add(signature)
            idx = len(chunks) + 1
            chunks.append(
                Chunk(
                    chunk_id=f"{source_id}:{idx}",
                    source_id=source_id,
                    title=title,
                    article=current_article,
                    text=content,
                    metadata={"strategy": "article_split"},
                )
            )

        for line in lines:
            m = self.article_pattern.match(line)
            if m:
                flush()
                current_lines = [line]
                current_article = f"Dieu {m.group(1)}"
            else:
                current_lines.append(line)
        flush()
        return chunks

    def _window_chunk(self, source_id: str, title: str, text: str) -> list[Chunk]:
        chunks: list[Chunk] = []
        idx: int = 1
        for content in self._slice_text(text):
            chunks.append(
                Chunk(
                    chunk_id=f"{source_id}:{idx}",
                    source_id=source_id,
                    title=title,
                    article=None,
                    text=content,
                    metadata={"strategy": "window"},
                )
            )
            idx += 1
        return chunks

    def _split_long_chunk(self, chunk: Chunk) -> list[Chunk]:
        split_chunks: list[Chunk] = []
        part: int = 1
        for content in self._slice_text(chunk.text):
            # Keep article anchor in every part to improve retrieval consistency.
            if chunk.article and not self.article_line_pattern.match(content):
                content = f"{chunk.article}\n{content}"
            metadata = dict(chunk.metadata)
            metadata["part"] = part
            split_chunks.append(
                Chunk(
                    chunk_id=f"{chunk.chunk_id}.p{part}",
                    source_id=chunk.source_id,
                    title=chunk.title,
                    article=chunk.article,
                    text=content,
                    metadata=metadata,
                )
            )
            part += 1
        return split_chunks

    def _slice_text(self, text: str) -> list[str]:
        outputs: list[str] = []
        start = 0
        n = len(text)
        while start < n:
            hard_end = min(start + self.max_chars, n)
            end = self._snap_end(text, start, hard_end)
            if end <= start:
                end = hard_end

            content = text[start:end].strip()
            if content:
                outputs.append(content)

            if end >= n:
                break

            next_start = max(end - self.overlap_chars, start + 1)
            next_start = self._snap_start(text, next_start, end)
            if next_start <= start:
                next_start = end
            start = next_start
        return outputs

    def _snap_end(self, text: str, start: int, hard_end: int) -> int:
        if hard_end >= len(text):
            return len(text)

        min_end = start + int(self.max_chars * 0.6)
        best = -1
        for sep in ("\n\n", "\n", ". ", "; ", ": ", ", ", " "):
            pos = text.rfind(sep, start, hard_end)
            if pos < 0:
                continue
            candidate = pos + len(sep)
            if candidate >= min_end:
                best = max(best, candidate)
        return best if best > 0 else hard_end

    def _snap_start(self, text: str, start: int, prev_end: int) -> int:
        n = len(text)
        s = max(0, min(start, n))

        while s < n and text[s].isspace():
            s += 1

        # Avoid starting in the middle of a token when overlap lands inside a word.
        if 0 < s < n and not text[s - 1].isspace():
            limit = min(n, s + 80)
            while s < limit and text[s].isalnum():
                s += 1
            while s < n and text[s].isspace():
                s += 1

        if s >= prev_end and prev_end < n:
            s = prev_end
        return s
