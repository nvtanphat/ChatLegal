from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

import httpx
from bs4 import BeautifulSoup
from loguru import logger

from src.database.models import LegalDocument
from src.database.mongo_client import MongoService
from src.settings import ROOT_DIR

VBPL_URL = "https://vbpl.vn/TW/Pages/vbpq-toanvan.aspx?ItemID={item_id}"
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0.0.0 Safari/537.36"
)
TITLE_SCRIPT_PATTERN = re.compile(r"var\s+title1\s*=\s*'([^']+)'", flags=re.IGNORECASE)
DOC_CODE_PATTERN = re.compile(r"\b(\d{1,3}/\d{4}/[^\s,;:()]+)\b", flags=re.IGNORECASE)

PRESET_ITEM_IDS: dict[str, tuple[str, ...]] = {
    "vbpl9": (
        "179420",  # Luat 85/2025/QH15
        "134278",  # Nghi quyet 01/2019/NQ-HDTP
        "16333",  # Nghi quyet 02/2004/NQ-HDTP
        "169032",  # Luat 27/2023/QH15 (Nha o)
        "161263",  # Luat 19/2023/QH15
        "172772",  # Nghi dinh 130/2024/ND-CP
        "165747",  # Quyet dinh 07/2024/QD-UBND
        "12400",  # Thong tu 03/2008/TT-BNG
        "18610",  # Quyet dinh 166/2004/QD-BCN
    ),
    "civil-focus-2024": (
        "95942",  # Bo luat 91/2015/QH13 (Dan su)
        "36870",  # Luat 52/2014/QH13 (Hon nhan va gia dinh)
        "177815",  # Luat 31/2024/QH15 (Dat dai)
        "169032",  # Luat 27/2023/QH15 (Nha o)
        "161263",  # Luat 19/2023/QH15 (Bao ve quyen loi nguoi tieu dung)
        "16333",  # Nghi quyet 02/2004/NQ-HDTP (thua ke)
        "134278",  # Nghi quyet 01/2019/NQ-HDTP (lai suat, hop dong)
        "47412",  # Nghi dinh 126/2014/ND-CP (huong dan Luat HN&GĐ)
        "169363",  # Nghi dinh 102/2024/ND-CP (huong dan Luat Dat dai)
    ),
    "bat-dong-san": (
        "169027",  # Luat Kinh doanh bat dong san 2023
        "177815",  # Luat Dat dai 2024
        "169032",  # Luat Nha o 2023
        "175927",  # Nghi dinh 96/2024/ND-CP (Huong dan Luat KDBDS)
        "175114",  # Nghi dinh 95/2024/ND-CP (Huong dan Luat Nha o)
    ),
}


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def fix_mojibake(value: str) -> str:
    if not value:
        return value
    try:
        return value.encode("latin1").decode("utf-8")
    except UnicodeError:
        return value


def parse_doc_code(text: str) -> str | None:
    candidates = [normalize_text(code).strip(".") for code in DOC_CODE_PATTERN.findall(text or "")]
    if not candidates:
        return None
    preferred_suffixes = ("NĐ-CP", "ND-CP", "NQ-HĐTP", "NQ-HDTP", "QH", "VBHN")
    for code in candidates:
        upper = code.upper()
        if any(suffix in upper for suffix in preferred_suffixes):
            return upper
    return candidates[0].upper()


def extract_title(item_id: str, html: str, soup: BeautifulSoup) -> str:
    script_match = TITLE_SCRIPT_PATTERN.search(html)
    if script_match:
        script_title = normalize_text(fix_mojibake(script_match.group(1)))
        if script_title:
            return script_title

    title_tag = soup.select_one("h1, h2, .title, .vbpq-title, title")
    if not title_tag:
        return f"VBPL Item {item_id}"
    title = normalize_text(fix_mojibake(title_tag.get_text(" ", strip=True)))
    return title or f"VBPL Item {item_id}"


def parse_vbpl_html(item_id: str, html: str) -> LegalDocument:
    soup = BeautifulSoup(html, "html.parser")
    title = extract_title(item_id=item_id, html=html, soup=soup)

    main = (
        soup.select_one("#toanvancontent")
        or soup.select_one("#divContent")
        or soup.select_one(".content")
        or soup.find("body")
        or soup
    )
    for tag in main(["script", "style", "noscript"]):
        tag.decompose()

    paragraphs = [normalize_text(fix_mojibake(node.get_text(" ", strip=True))) for node in main.select("p, div, h1, h2, h3, h4, h5, h6, b")]
    text = "\n".join([line for line in paragraphs if line])
    if not text:
        text = normalize_text(fix_mojibake(main.get_text("\n", strip=True)))

    return LegalDocument(
        source="vbpl.vn",
        source_id=str(item_id),
        title=title,
        doc_code=parse_doc_code(title) or parse_doc_code(text),
        text=text,
        raw_html=html,
        metadata={"url": VBPL_URL.format(item_id=item_id)},
    )


def fetch_doc(item_id: str, client: httpx.Client) -> LegalDocument:
    url = VBPL_URL.format(item_id=item_id)
    logger.info("Fetching {}", url)
    response = client.get(url)
    response.raise_for_status()
    return parse_vbpl_html(item_id=item_id, html=response.text)


def save_raw_docs(docs: Iterable[LegalDocument], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for doc in docs:
        path = out_dir / f"{doc.source_id}.json"
        path.write_text(
            json.dumps(doc.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("Saved {}", path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crawl Vietnamese legal docs from vbpl.vn")
    parser.add_argument("--item-id", action="append", default=[], help="VBPL ItemID, can repeat")
    parser.add_argument(
        "--preset",
        action="append",
        choices=sorted(PRESET_ITEM_IDS.keys()),
        default=[],
        help="Predefined ItemID bundle. Can repeat.",
    )
    parser.add_argument("--list-presets", action="store_true", help="Show available presets and exit.")
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    parser.add_argument(
        "--output-dir",
        default=str(ROOT_DIR / "data" / "raw" / "legal_docs"),
        help="Directory to write raw JSON docs",
    )
    parser.add_argument("--to-mongo", action="store_true", help="Insert docs into MongoDB")
    return parser.parse_args()


def merge_item_ids(item_ids: list[str], presets: list[str]) -> list[str]:
    merged: list[str] = []
    for preset in presets:
        merged.extend(PRESET_ITEM_IDS[preset])
    merged.extend(item_ids)

    seen: set[str] = set()
    unique: list[str] = []
    for item in merged:
        item_id = str(item).strip()
        if not item_id or item_id in seen:
            continue
        seen.add(item_id)
        unique.append(item_id)
    return unique


def print_presets() -> None:
    logger.info("Available presets:")
    for name in sorted(PRESET_ITEM_IDS):
        logger.info("- {}: {}", name, ", ".join(PRESET_ITEM_IDS[name]))


def main() -> None:
    args = parse_args()
    if args.list_presets:
        print_presets()
        return

    direct_item_ids = [str(item).strip() for item in args.item_id if str(item).strip()]
    item_ids = merge_item_ids(item_ids=direct_item_ids, presets=args.preset)
    if not item_ids:
        raise SystemExit("Missing --item-id/--preset. Example: --preset civil-focus-2024")

    docs: list[LegalDocument] = []
    with httpx.Client(
        timeout=args.timeout,
        follow_redirects=True,
        headers={"User-Agent": args.user_agent.strip() or DEFAULT_USER_AGENT},
    ) as client:
        for item_id in item_ids:
            try:
                docs.append(fetch_doc(item_id=item_id, client=client))
            except Exception as exc:
                logger.error("Failed ItemID {}: {}", item_id, exc)

    if not docs:
        raise SystemExit("No documents fetched successfully.")

    save_raw_docs(docs=docs, out_dir=Path(args.output_dir))

    if args.to_mongo:
        mongo = MongoService()
        mongo.connect()
        inserted = mongo.insert_legal_docs(docs)
        logger.info("Inserted {} docs into MongoDB.", inserted)


if __name__ == "__main__":
    main()
