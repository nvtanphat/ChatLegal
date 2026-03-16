from __future__ import annotations

import json
import re
import unicodedata
from typing import Any

from src.core.base_llm import BaseLLM
from src.settings import load_settings


class QueryReflector:
    _ws_pattern = re.compile(r"\s+")
    _article_pattern = re.compile(r"\bdieu\s+(\d+[a-zA-Z0-9]*)\b", flags=re.IGNORECASE)
    _clause_pattern = re.compile(r"\bkhoan\s+(\d+[a-zA-Z0-9]*)\b", flags=re.IGNORECASE)
    _doc_code_pattern = re.compile(r"\b(\d{1,3}/\d{4}/[A-Za-z0-9\-]+)\b", flags=re.IGNORECASE)
    _item_id_pattern = re.compile(r"\bitem\s*id\s*[:=]?\s*(\d{4,9})\b", flags=re.IGNORECASE)

    _abbr_map = {
        "blds": "bo luat dan su",
        "bl": "bo luat",
        "hngd": "hon nhan va gia dinh",
        "qsd": "quyen su dung",
        "bds": "bat dong san",
    }

    def __init__(self, llm: BaseLLM | None = None, use_llm_reflect: bool | None = None) -> None:
        retrieval_cfg = load_settings().get("retrieval", {})
        reflect_cfg = retrieval_cfg.get("query_reflection", {}) or {}
        multi_query_cfg = retrieval_cfg.get("multi_query", {}) or {}
        hyde_cfg = retrieval_cfg.get("hyde", {}) or {}
        metadata_filter_cfg = retrieval_cfg.get("metadata_filter", {}) or {}

        self.llm = llm
        self.use_llm_reflect = (
            bool(reflect_cfg.get("use_llm", False)) if use_llm_reflect is None else use_llm_reflect
        )
        self.multi_query_enabled = bool(multi_query_cfg.get("enabled", True))
        self.multi_query_variants = max(0, int(multi_query_cfg.get("variants", 2)))
        self.multi_query_use_llm = bool(
            multi_query_cfg.get("use_llm", self.use_llm_reflect),
        )
        self.hyde_enabled = bool(hyde_cfg.get("enabled", False))
        self.hyde_max_chars = max(64, int(hyde_cfg.get("max_chars", 500)))
        self.metadata_filter_enabled = bool(metadata_filter_cfg.get("enabled", True))

    def rewrite(self, query: str) -> str:
        normalized = self._normalize_query(query)
        if not self.use_llm_reflect or self.llm is None:
            return normalized

        prompt = (
            "Viet lai cau hoi phap ly ngan gon, ro nghia, giu nguyen y dinh nguoi dung.\n"
            f"Cau hoi goc: {normalized}\n"
            "Cau hoi viet lai:"
        )
        try:
            rewritten = self.llm.generate(prompt=prompt, temperature=0.0)
            rewritten = self._ws_pattern.sub(" ", rewritten).strip()
            return rewritten or normalized
        except Exception:
            return normalized

    def expand_queries(self, query: str) -> list[str]:
        normalized = self._normalize_query(query)
        if not normalized or not self.multi_query_enabled or self.multi_query_variants <= 0:
            return []

        variants: list[str] = []
        if self.multi_query_use_llm and self.llm is not None:
            variants.extend(self._expand_queries_with_llm(normalized))
        variants.extend(self._expand_queries_heuristic(normalized))

        outputs: list[str] = []
        seen: set[str] = {normalized.casefold()}
        for item in variants:
            item = self._normalize_query(item)
            if not item:
                continue
            item_key = item.casefold()
            if item_key in seen:
                continue
            outputs.append(item)
            seen.add(item_key)
            if len(outputs) >= self.multi_query_variants:
                break
        return outputs

    def build_hyde(self, query: str) -> str | None:
        if not self.hyde_enabled or self.llm is None:
            return None
        normalized = self._normalize_query(query)
        if not normalized:
            return None

        prompt = (
            "Viet 1 doan tra loi gia dinh ngan gon (2-4 cau) cho cau hoi phap ly sau "
            "de phuc vu truy xuat tai lieu. "
            "Khong them canh bao, khong giai thich dai.\n"
            f"Cau hoi: {normalized}\n"
            "Doan tra loi gia dinh:"
        )
        try:
            hyde_text = self.llm.generate(prompt=prompt, temperature=0.0)
            hyde_text = self._ws_pattern.sub(" ", hyde_text).strip()
            if not hyde_text:
                return None
            return hyde_text[: self.hyde_max_chars]
        except Exception:
            return None

    def extract_metadata_filter(self, query: str) -> dict[str, Any] | None:
        if not self.metadata_filter_enabled:
            return None

        normalized = self._normalize_query(query)
        if not normalized:
            return None

        plain = self._strip_accents(normalized).lower()
        filters: list[dict[str, Any]] = []

        article_match = self._article_pattern.search(plain)
        if article_match:
            article_num = article_match.group(1)
            article_variants = [f"Dieu {article_num}", f"Điều {article_num}"]
            filters.append(
                {"$or": [{"article": {"$eq": article}} for article in article_variants]}
            )

        code_match = self._doc_code_pattern.search(normalized)
        if code_match:
            filters.append({"doc_code": {"$eq": code_match.group(1).upper()}})

        item_id_match = self._item_id_pattern.search(plain)
        if item_id_match:
            filters.append({"source_id": {"$eq": item_id_match.group(1)}})

        if not filters:
            return None
        if len(filters) == 1:
            return filters[0]
        return {"$and": filters}

    def _normalize_query(self, query: str) -> str:
        normalized = self._ws_pattern.sub(" ", (query or "").strip())
        for key, value in self._abbr_map.items():
            normalized = re.sub(
                rf"\b{re.escape(key)}\b",
                value,
                normalized,
                flags=re.IGNORECASE,
            )
        return self._ws_pattern.sub(" ", normalized).strip()

    def _expand_queries_with_llm(self, query: str) -> list[str]:
        prompt = (
            "Tao cac truy van tim kiem de truy xuat van ban phap luat dan su Viet Nam.\n"
            f"Yeu cau: tra ve JSON array toi da {self.multi_query_variants} chuoi, "
            "moi chuoi ngan gon va giu nguyen y dinh.\n"
            f"Cau hoi goc: {query}\n"
            "JSON array:"
        )
        try:
            raw = self.llm.generate(prompt=prompt, temperature=0.0)
        except Exception:
            return []

        raw = raw.strip()
        if not raw:
            return []

        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(item) for item in parsed if str(item).strip()]
        except Exception:
            pass

        lines = []
        for item in raw.splitlines():
            item = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", item).strip()
            if item:
                lines.append(item)
        return lines

    def _expand_queries_heuristic(self, query: str) -> list[str]:
        plain = self._strip_accents(query).lower()
        outputs: list[str] = []

        article_match = self._article_pattern.search(plain)
        clause_match = self._clause_pattern.search(plain)
        code_match = self._doc_code_pattern.search(query)
        if article_match:
            article = article_match.group(1)
            if code_match:
                outputs.append(f"Dieu {article} {code_match.group(1).upper()} quy dinh gi")
            if clause_match:
                outputs.append(
                    f"khoan {clause_match.group(1)} dieu {article} quy dinh nhu the nao",
                )
            outputs.append(f"noi dung dieu {article} bo luat dan su")

        if "hop dong" in plain:
            outputs.append(f"quy dinh bo luat dan su ve {query}")
        if "thua ke" in plain:
            outputs.append("thua ke theo phap luat bo luat dan su quy dinh ra sao")
        if "dat" in plain or "nha" in plain:
            outputs.append(f"quyen va nghia vu lien quan {query}")

        return outputs

    @staticmethod
    def _strip_accents(text: str) -> str:
        decomposed = unicodedata.normalize("NFKD", text or "")
        return "".join(char for char in decomposed if not unicodedata.combining(char))
