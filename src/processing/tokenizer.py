from __future__ import annotations

import re

_word_pattern = re.compile(r"\w+", re.UNICODE)

try:
    from underthesea import word_tokenize
except ImportError:  # pragma: no cover
    word_tokenize = None


def tokenize_vi(text: str) -> list[str]:
    normalized = (text or "").strip()
    if not normalized:
        return []

    if word_tokenize is not None:
        tokens = word_tokenize(normalized, format="text").split()
        return [tok.lower() for tok in tokens]

    return [tok.lower() for tok in _word_pattern.findall(normalized)]
