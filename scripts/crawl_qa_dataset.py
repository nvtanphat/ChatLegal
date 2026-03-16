from __future__ import annotations

import argparse
import html
import json
import logging
import random
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup, Tag
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.settings import ROOT_DIR

BASE_TVPL = "https://thuvienphapluat.vn"
DEFAULT_MAX_PAGES = 30
DEFAULT_TARGET_COUNT = 1200
OUTPUT_FILE = ROOT_DIR / "data" / "raw" / "qa_pairs" / "qa_pairs_raw.json"
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0",
    "Mozilla/5.0 (AppleWebKit/537.36, like Gecko) Chrome/121.0.0.0 Safari/537.36",
]
DEFAULT_USER_AGENT = random.choice(USER_AGENTS)
SEARCH_URL = "https://thuvienphapluat.vn/hoi-dap-phap-luat/tim-kiem?keyword={keyword}"
DETAIL_LINK_PATTERN = re.compile(r"/hoi-dap-phap-luat/.+-\d+\.html$", flags=re.IGNORECASE)
DEFAULT_TOPICS = (
    "hoi-dap-phap-luat-moi-nhat",
    "quyen-dan-su",
    "hon-nhan-gia-dinh",
    "bat-dong-san",
    "thua-ke",
    "thu-tuc-ly-hon",
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class QATopic:
    slug: str

    @property
    def listing_url(self) -> str:
        return f"{BASE_TVPL}/hoi-dap-phap-luat/{self.slug}"


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(value or "")).strip()


def build_session(
    retry_total: int = 6,
    retry_connect: int = 5,
    retry_read: int = 3,
    retry_status: int = 2,
    retry_backoff: float = 1.0,
    user_agent: str | None = None,
    cookie: str = "",
) -> requests.Session:
    session = requests.Session()
    ua = user_agent or random.choice(USER_AGENTS)
    retry = Retry(
        total=max(0, retry_total),
        connect=max(0, retry_connect),
        read=max(0, retry_read),
        status=max(0, retry_status),
        backoff_factor=max(0.0, retry_backoff),
        status_forcelist=[403, 429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update(
        {
            "User-Agent": user_agent.strip() or DEFAULT_USER_AGENT,
            "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.6,en;q=0.4",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Referer": BASE_TVPL,
        }
    )
    if cookie.strip():
        session.headers["Cookie"] = cookie.strip()
    session.headers["User-Agent"] = ua
    return session


class TVPLRawCrawler:
    def __init__(
        self,
        topics: tuple[QATopic, ...],
        start_page: int = 1,
        max_pages: int = DEFAULT_MAX_PAGES,
        target_count: int = DEFAULT_TARGET_COUNT,
        delay_min: float = 1.5,
        delay_max: float = 3.5,
        timeout: int = 35,
        retry_total: int = 6,
        retry_connect: int = 5,
        retry_read: int = 3,
        retry_status: int = 2,
        retry_backoff: float = 1.0,
        user_agent: str | None = None,
        cookie: str = "",
        show_progress: bool = False,
        progress_writer: Callable[[str], None] | None = None,
        keywords: list[str] | None = None,
    ) -> None:
        self.topics = topics
        self.keywords = keywords or []
        self.start_page = max(1, start_page)
        self.max_pages = max_pages
        self.target_count = target_count
        self.delay_min = delay_min
        self.delay_max = delay_max
        self.timeout = timeout
        self.session = build_session(
            retry_total=retry_total,
            retry_connect=retry_connect,
            retry_read=retry_read,
            retry_status=retry_status,
            retry_backoff=retry_backoff,
            user_agent=user_agent,
            cookie=cookie,
        )
        self.seen_urls: set[str] = set()
        self.show_progress = show_progress
        self.progress_writer = progress_writer or (lambda text: sys.stdout.write(text))

    def _progress(self: TVPLRawCrawler, prefix: str, current: int, total: int, final: bool = False) -> None:
        if not self.show_progress:
            return
        safe_total = max(1, total)
        clamped = min(max(current, 0), safe_total)
        width = 28
        done = int(width * clamped / safe_total)
        bar = f"[{'#' * done}{'.' * (width - done)}] {clamped}/{safe_total}"
        self.progress_writer(f"\r{prefix} {bar}")
        if final:
            self.progress_writer("\n")

    def random_sleep(self) -> None:
        time.sleep(random.uniform(self.delay_min, self.delay_max))

    def get_page(self: TVPLRawCrawler, url: str) -> requests.Response | None:
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            response.encoding = "utf-8"
            return response
        except requests.RequestException as exc:
            LOGGER.error("Request failed for %s: %s", url, exc)
            return None

    def crawl(self: TVPLRawCrawler) -> list[dict]:
        all_links: list[tuple[str, str]] = []
        for topic in self.topics:
            links = self.get_article_links(topic.listing_url, f"Topic={topic.slug}")
            all_links.extend((topic.slug, link) for link in links)
            LOGGER.info("Topic=%s total_links=%s", topic.slug, len(links))
        
        for kw in self.keywords:
            search_url = SEARCH_URL.format(keyword=kw)
            links = self.get_article_links(search_url, f"Keyword={kw}")
            all_links.extend((f"search:{kw}", link) for link in links)
            LOGGER.info("Keyword=%s total_links=%s", kw, len(links))

        items: list[dict] = []
        crawl_total = min(len(all_links), self.target_count)
        if crawl_total == 0:
            self._progress("Crawling QA", 0, 1, final=True)
        for idx, item in enumerate(all_links, start=1):
            topic_slug, url = item
            if len(items) >= self.target_count:
                break
            if url in self.seen_urls:
                self._progress("Crawling QA", min(idx, crawl_total), max(1, crawl_total))
                continue
            self.seen_urls.add(str(url))

            item = self.parse_article(url=str(url), topic_slug=str(topic_slug))
            if item is not None:
                items.append(item)
                if len(items) % 25 == 0:
                    LOGGER.info("Raw crawl progress: %s/%s items (links=%s)", len(items), self.target_count, idx)
            self._progress("Crawling QA", min(idx, crawl_total), max(1, crawl_total))
            self.random_sleep()

        self._progress("Crawling QA", crawl_total, max(1, crawl_total), final=True)
        return items

    def get_article_links(self: TVPLRawCrawler, base_url: str, label: str) -> list[str]:
        links: list[str] = []
        no_new_pages: int = 0
        end_page = self.start_page + self.max_pages
        total_pages = max(1, end_page - self.start_page)
        for page_idx, page_number in enumerate(range(self.start_page, end_page), start=1):
            sep = "&" if "?" in base_url else "?"
            url = base_url if page_number == 1 else f"{base_url}{sep}page={page_number}"
            response = self.get_page(url)
            if response is None:
                self._progress(f"Listing {label}", page_idx, total_pages)
                self.random_sleep()
                continue

            soup = BeautifulSoup(response.text, "html.parser")
            before = len(links)
            for anchor in soup.find_all("a", href=True):
                href = normalize_text(anchor.get("href", ""))
                if not DETAIL_LINK_PATTERN.search(href):
                    continue
                full_url = href if href.startswith("http") else urljoin(BASE_TVPL, href)
                if full_url not in links:
                    links.append(full_url)

            new_count = len(links) - before
            LOGGER.info("%s page=%s new_links=%s total=%s", label, page_number, new_count, len(links))

            if new_count == 0:
                no_new_pages = int(no_new_pages) + 1
            else:
                no_new_pages = 0
            if no_new_pages >= 3:
                self._progress(f"Listing {label}", page_idx, total_pages, final=True)
                break
            self._progress(f"Listing {label}", page_idx, total_pages)
            self.random_sleep()
        else:
            self._progress(f"Listing {label}", total_pages, total_pages, final=True)
        return links

    @staticmethod
    def find_content_container(soup: BeautifulSoup) -> Tag | None:
        selectors = (
            "article",
            "div.content",
            "div.fck_detail",
            "div.article-content",
            "div.news-content",
            "div.main-content",
            "div#content",
        )
        for selector in selectors:
            node = soup.select_one(selector)
            if node and len(normalize_text(node.get_text(" ", strip=True))) > 250:
                return node
        return soup.find("article") or soup.find("body")

    @staticmethod
    def collect_text_lines(container: Tag) -> list[str]:
        for tag in container(["script", "style", "form", "noscript"]):
            tag.decompose()

        lines: list[str] = []
        seen: set[str] = set()
        for node in container.find_all(["p", "li", "blockquote", "h2", "h3", "h4"]):
            text = normalize_text(node.get_text(" ", strip=True))
            if len(text) < 20:
                continue
            if text in seen:
                continue
            seen.add(text)
            lines.append(text)
            if len(lines) >= 160:
                break
        return lines

    def parse_article(self: TVPLRawCrawler, url: str, topic_slug: str) -> dict | None:
        response = self.get_page(url)
        if response is None:
            return None

        soup = BeautifulSoup(response.text, "html.parser")
        title_element = soup.find("h1")
        if not title_element:
            return None

        question = normalize_text(title_element.get_text(" ", strip=True))
        if len(question) < 12:
            return None

        container = self.find_content_container(soup)
        if container is None:
            return None
        lines = self.collect_text_lines(container)
        answer_raw = "\n".join(lines).strip()
        if answer_raw.casefold().startswith(question.casefold()):
            answer_raw = answer_raw[len(question) :].lstrip(":?- \n")
        if len(answer_raw) < 60:
            return None

        return {
            "source": "tvpl",
            "source_site": "thuvienphapluat.vn",
            "topic": topic_slug,
            "question": question,
            "answer_raw": answer_raw,
            "url": url,
            "crawled_at": datetime.now().isoformat(timespec="seconds"),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crawl raw QA items from thuvienphapluat.vn")
    parser.add_argument("--topic", action="append", help="Topic slug under /hoi-dap-phap-luat/<slug>")
    parser.add_argument("--keyword", action="append", help="Search keyword (e.g. law code)")
    parser.add_argument("--start-page", type=int, default=1)
    parser.add_argument("--max-pages", type=int, default=DEFAULT_MAX_PAGES)
    parser.add_argument("--target-count", type=int, default=DEFAULT_TARGET_COUNT)
    parser.add_argument("--delay-min", type=float, default=1.5)
    parser.add_argument("--delay-max", type=float, default=3.5)
    parser.add_argument("--timeout", type=int, default=35)
    parser.add_argument("--retry-total", type=int, default=6)
    parser.add_argument("--retry-connect", type=int, default=5)
    parser.add_argument("--retry-read", type=int, default=3)
    parser.add_argument("--retry-status", type=int, default=2)
    parser.add_argument("--retry-backoff", type=float, default=1.0)
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    parser.add_argument("--cookie", default="")
    parser.set_defaults(show_progress=True)
    parser.add_argument("--show-progress", dest="show_progress", action="store_true")
    parser.add_argument("--no-progress", dest="show_progress", action="store_false")
    parser.add_argument("--output-file", default=str(OUTPUT_FILE))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    topic_slugs = tuple(dict.fromkeys(args.topic or list(DEFAULT_TOPICS)))
    topics = tuple(QATopic(slug=slug) for slug in topic_slugs)
    if not topics:
        raise SystemExit("No topics selected.")

    LOGGER.info(
        "Start raw crawl topics=%s start_page=%s max_pages=%s target_count=%s timeout=%s retry_total=%s retry_read=%s",
        ",".join(topic_slugs),
        args.start_page,
        args.max_pages,
        args.target_count,
        args.timeout,
        args.retry_total,
        args.retry_read,
    )
    crawler = TVPLRawCrawler(
        topics=topics,
        start_page=args.start_page,
        max_pages=args.max_pages,
        target_count=args.target_count,
        delay_min=args.delay_min,
        delay_max=args.delay_max,
        timeout=args.timeout,
        retry_total=args.retry_total,
        retry_connect=args.retry_connect,
        retry_read=args.retry_read,
        retry_status=args.retry_status,
        retry_backoff=args.retry_backoff,
        user_agent=args.user_agent,
        cookie=args.cookie,
        show_progress=args.show_progress,
        keywords=args.keyword,
    )
    items = crawler.crawl()
    if not items:
        raise SystemExit("No raw QA extracted.")

    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Saved %s raw QA items to %s", len(items), output_file)


if __name__ == "__main__":
    main()
