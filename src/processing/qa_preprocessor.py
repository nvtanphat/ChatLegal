from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Callable
from typing import Any

LAW_NAME_PATTERN = re.compile(
    r"(Bộ luật|Luật|Nghị định|Nghị quyết|Thông tư)\s+[^\n]{0,140}?\d{4}",
    flags=re.IGNORECASE,
)
DOC_CODE_PATTERN = re.compile(r"\b\d{1,4}/\d{2,4}/[A-ZĐa-z0-9\-]+\b")
ARTICLE_PATTERN = re.compile(
    r"(Điểm\s+[a-z]\s+Khoản\s+\d+\s+Điều\s+\d+[a-z]?|Khoản\s+\d+\s+Điều\s+\d+[a-z]?|Điều\s+\d+[a-z]?)",
    flags=re.IGNORECASE,
)
LEADING_QUOTE_PATTERN = re.compile(r"^(?:>{1,3}\s*)+")
IMAGE_NOTE_PATTERN = re.compile(r"\s*\((?:hình|image|ảnh)[^)]*\)\s*$", flags=re.IGNORECASE)
QUESTION_PREFIX_PATTERN = re.compile(r"^(?:câu hỏi|hỏi)\s*[:\-]", flags=re.IGNORECASE)
SUMMARY_LINE_PATTERN = re.compile(
    r"^\*?\s*trên đây là (?:nội dung|thông tin|bài viết).*$",
    flags=re.IGNORECASE,
)
FOLLOWUP_LINE_PATTERN = re.compile(r"^quý khách cần hỏi thêm thông tin.*$", flags=re.IGNORECASE)
RELATED_HINT_PATTERN = re.compile(r"^>{2,3}\s*.*\?\s*$")

CIVIL_KEYWORDS = {
    "dân sự",
    "hợp đồng",
    "giao dịch dân sự",
    "thừa kế",
    "di chúc",
    "bồi thường thiệt hại",
    "quyền sở hữu",
    "tài sản",
    "hôn nhân",
    "gia đình",
    "ly hôn",
    "con chung",
    "công chứng",
    "đất đai",
    "nhà ở",
    "bất động sản",
    "tranh chấp dân sự",
    "nghĩa vụ dân sự",
    "di sản",
    "chia tài sản",
    "ủy quyền",
    "uỷ quyền",
    "đặt cọc",
    "chuyển nhượng",
    "tặng cho",
    "giám hộ",
    "năng lực hành vi dân sự",
    "quyền sử dụng đất",
    "thế chấp",
    "bảo lãnh",
    "cầm cố",
    "sổ đỏ",
}
EXCLUDE_KEYWORDS = {
    "hình sự",
    "vi phạm hành chính",
    "hành chính",
    "thuế",
    "bảo hiểm xã hội",
    "lao động",
    "doanh nghiệp",
    "đầu tư",
    "hải quan",
    "giao thông",
    "nghĩa vụ quân sự",
    "tạm trú",
    "thường trú",
    "tố tụng",
    "khởi tố",
}
BASE_LAW_HINTS = {
    "bộ luật dân sự",
    "luật hôn nhân và gia đình",
    "luật đất đai",
    "quyền dân sự",
    "hợp đồng dân sự",
    "thừa kế",
    "quyền sử dụng đất",
}


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(value or "")).strip()


def _simplify_for_match(value: str) -> str:
    cleaned = normalize_text(value).casefold()
    cleaned = LEADING_QUOTE_PATTERN.sub("", cleaned)
    cleaned = IMAGE_NOTE_PATTERN.sub("", cleaned)
    cleaned = re.sub(r"[\W_]+", "", cleaned, flags=re.UNICODE)
    return cleaned


def _is_question_like_line(value: str) -> bool:
    cleaned = normalize_text(value)
    cleaned = LEADING_QUOTE_PATTERN.sub("", cleaned).strip()
    if not cleaned:
        return False
    if QUESTION_PREFIX_PATTERN.match(cleaned):
        return True
    return cleaned.endswith("?")


def clean_answer_text(value: str, *, question: str = "") -> str:
    lines = [normalize_text(line) for line in str(value or "").splitlines()]
    lines = [line for line in lines if line]

    question_key = _simplify_for_match(question)
    filtered_lines: list[str] = []
    for line in lines:
        line_key = _simplify_for_match(line)
        if question_key and question_key in line_key:
            continue
        stripped = line.strip()
        if SUMMARY_LINE_PATTERN.match(stripped):
            continue
        if FOLLOWUP_LINE_PATTERN.match(stripped):
            continue
        if RELATED_HINT_PATTERN.match(stripped):
            continue
        filtered_lines.append(line)
    lines = filtered_lines

    # Drop question suggestions often injected at the beginning of scraped answers.
    removed = 0
    while lines and removed < 6 and _is_question_like_line(lines[0]):
        lines.pop(0)
        removed += 1

    deduped: list[str] = []
    seen: set[str] = set()
    for line in lines:
        lowered = line.casefold()
        if lowered in seen:
            continue
        seen.add(lowered)
        deduped.append(line)
    return "\n".join(deduped).strip()


def load_law_signals(legal_docs_dir: Path, extra_keywords: list[str] | None = None) -> tuple[set[str], set[str]]:
    keywords: set[str] = set(BASE_LAW_HINTS)
    doc_codes: set[str] = set()

    if legal_docs_dir.exists():
        for path in sorted(legal_docs_dir.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue

            title = normalize_text(str(payload.get("title") or ""))
            if title:
                keywords.add(title.casefold())

            doc_code = normalize_text(str(payload.get("doc_code") or ""))
            if doc_code:
                doc_codes.add(doc_code.casefold())

            text = normalize_text(str(payload.get("text") or ""))[:140000]
            for code in DOC_CODE_PATTERN.findall(text):
                doc_codes.add(normalize_text(code).casefold())
            for match in LAW_NAME_PATTERN.finditer(text):
                law_name = normalize_text(match.group(0))
                if 8 <= len(law_name) <= 160:
                    keywords.add(law_name.casefold())

    if extra_keywords:
        for item in extra_keywords:
            cleaned = normalize_text(item)
            if cleaned:
                keywords.add(cleaned.casefold())

    return keywords, doc_codes


def count_keyword_hits(text: str, keywords: set[str]) -> int:
    lowered = text.casefold()
    return sum(1 for keyword in keywords if keyword in lowered)


def is_relevant(
    question: str,
    answer: str,
    law_keywords: set[str],
    doc_codes: set[str],
    strict_related: bool = True,
) -> bool:
    combined = f"{question}\n{answer}"
    lowered = combined.casefold()

    law_hits = count_keyword_hits(combined, law_keywords)
    code_hits = sum(1 for code in doc_codes if code in lowered)
    civil_hits = count_keyword_hits(combined, CIVIL_KEYWORDS)
    exclude_hits = count_keyword_hits(combined, EXCLUDE_KEYWORDS)

    if strict_related:
        if law_hits + code_hits == 0:
            return False
        return exclude_hits <= civil_hits + law_hits + code_hits

    if law_hits + code_hits > 0:
        return True
    if exclude_hits > civil_hits + 1:
        return False
    return civil_hits >= 2


def extract_cited_articles(text: str) -> list[str]:
    results: list[str] = []
    for match in ARTICLE_PATTERN.finditer(text):
        value = normalize_text(match.group(0))
        if value not in results:
            results.append(value)
    return results


def extract_cited_laws(text: str, law_keywords: set[str]) -> list[str]:
    lowered = text.casefold()
    results: list[str] = []
    for keyword in sorted(law_keywords):
        if keyword in lowered:
            results.append(keyword)
    return results[:20]


def preprocess_qa_items(
    raw_items: list[dict[str, Any]],
    law_keywords: set[str],
    doc_codes: set[str],
    target_count: int = 500,
    strict_related: bool = True,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[dict[str, Any]]:
    processed: list[dict[str, Any]] = []
    seen_questions: set[str] = set()
    seen_urls: set[str] = set()
    total = len(raw_items)

    for idx, item in enumerate(raw_items, start=1):
        if len(processed) >= target_count:
            break

        if progress_callback is not None:
            progress_callback(idx, total)

        question = normalize_text(str(item.get("question") or ""))
        answer = clean_answer_text(
            str(item.get("answer_raw") or item.get("answer") or ""),
            question=question,
        )
        url = normalize_text(str(item.get("url") or ""))
        topic = normalize_text(str(item.get("topic") or "other"))
        if len(question) < 12 or len(answer) < 120 or not url:
            continue
        q_key = question.casefold()
        if q_key in seen_questions or url in seen_urls:
            continue

        if not is_relevant(question, answer, law_keywords, doc_codes, strict_related=strict_related):
            continue

        combined = f"{question}\n{answer}"
        processed.append(
            {
                "source": "tvpl",
                "source_site": "thuvienphapluat.vn",
                "scope": "civil",
                "topic": topic,
                "question": question,
                "answer": answer,
                "url": url,
                "tags": ["dan-su", topic],
                "cited_laws": extract_cited_laws(combined, law_keywords),
                "cited_articles": extract_cited_articles(combined),
                "crawled_at": normalize_text(str(item.get("crawled_at") or "")),
            }
        )
        seen_questions.add(q_key)
        seen_urls.add(url)

    if progress_callback is not None and total > 0 and len(processed) < target_count:
        progress_callback(total, total)

    return processed
