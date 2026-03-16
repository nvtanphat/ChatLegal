from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(slots=True)
class IntentResult:
    intent: str
    confidence: float
    reason: str


class IntentRouter:
    legal_keywords = (
        "luat", "dieu", "bl", "bo luat", "hop dong", "thua ke", "tai san", 
        "so huu", "dat dai", "hon nhan", "ket hon", "ly hon", "gia dinh", 
        "nghia vu", "quyen", "trach nghiem", "phap luat", "dan su", "bao nhiu",
        "quy dinh", "tuoi", "phat", "vi pham"
    )
    chitchat_keywords = (
        "thoi tiet", "mua", "nang", "chao", "hello", "xin chao", "cam on", 
        "ban ten gi", "la ai", "khoe khong"
    )
    out_of_scope_keywords = ("to tung", "khoi kien", "toa an", "blttds", "hinh su")
    article_pattern = re.compile(r"\bdieu\s+\d+", flags=re.IGNORECASE)

    @staticmethod
    def strip_accents(text: str) -> str:
        import unicodedata
        nfkd_form = unicodedata.normalize('NFKD', text)
        return "".join([c for c in nfkd_form if not unicodedata.combining(c)])

    def route(self, query: str) -> IntentResult:
        normalized = (query or "").strip().lower()
        if not normalized:
            return IntentResult(intent="chitchat", confidence=0.3, reason="empty_query")
        
        # Use a version without accents for better keyword matching
        plain_text = self.strip_accents(normalized)

        if any(token in plain_text for token in self.out_of_scope_keywords):
            return IntentResult(intent="out_of_scope", confidence=0.95, reason="procedure_law")

        if self.article_pattern.search(plain_text):
            return IntentResult(intent="legal_query", confidence=0.95, reason="article_lookup")

        if any(token in plain_text for token in self.chitchat_keywords):
            return IntentResult(intent="chitchat", confidence=0.8, reason="small_talk")

        if any(token in plain_text for token in self.legal_keywords):
            return IntentResult(intent="legal_query", confidence=0.75, reason="legal_keyword")

        return IntentResult(intent="chitchat", confidence=0.55, reason="default")

