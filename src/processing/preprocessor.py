from __future__ import annotations

import html
import re


class LegalPreprocessor:
    _html_tag_pattern = re.compile(r"<[^>]+>")
    _noise_line_patterns = (
        re.compile(r"^\s*(thuộc tính|lịch sử|vb liên quan|lược đồ|bản pdf|tải về)\s*$", re.IGNORECASE),
        re.compile(r"cơ sở dữ liệu (quốc gia|văn bản pháp luật)", re.IGNORECASE),
        re.compile(r"csdl (trung ương|quốc gia)", re.IGNORECASE),
        re.compile(r"^văn bản quy phạm pháp luật\b", re.IGNORECASE),
    )

    def clean_text(self, text: str) -> str:
        text = html.unescape(text or "")
        text = self._html_tag_pattern.sub(" ", text)
        text = text.replace("\u00a0", " ")
        # Normalize non-newline whitespaces to a single space
        text = re.sub(r"[ \t\f\v]+", " ", text)
        # Normalize multiple newlines to a single newline and strip whitespace around them
        text = re.sub(r" ?\n ?", "\n", text)
        text = re.sub(r"\n+", "\n", text)
        text = self._filter_noise_lines(text)
        return text.strip()

    def _filter_noise_lines(self, text: str) -> str:
        lines = [line.strip() for line in text.splitlines()]
        outputs: list[str] = []
        for line in lines:
            if not line:
                continue
            if self._is_noise_line(line):
                continue
            if outputs and line == outputs[-1]:
                continue
            outputs.append(line)
        return "\n".join(outputs)

    def _is_noise_line(self, line: str) -> bool:
        lowered = line.lower()
        if lowered.startswith("http://") or lowered.startswith("https://"):
            return True
        for pattern in self._noise_line_patterns:
            if pattern.search(line):
                return True
        return False
