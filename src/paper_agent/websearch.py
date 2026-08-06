"""联网论文搜索：arXiv API（免费、无需 key）。

查询词建议用英文（arXiv 论文以英文为主）。遵守 arXiv API 限速：请求间隔 ≥3 秒。
"""
import threading
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Optional

ARXIV_API = "https://export.arxiv.org/api/query"
_ATOM = "{http://www.w3.org/2005/Atom}"
_TIMEOUT = 20
_MIN_INTERVAL = 3.0  # arXiv 官方要求 ≥3s/请求

_last_request: float = 0.0
_lock = threading.Lock()


class WebSearchError(Exception):
    pass


@dataclass
class WebPaper:
    title: str
    authors: list[str]
    year: Optional[int]
    abstract: str
    url: str  # arXiv 摘要页
    pdf_url: Optional[str]


def _throttle() -> None:
    global _last_request
    with _lock:
        now = time.monotonic()
        wait = _last_request + _MIN_INTERVAL - now
        if wait > 0:
            time.sleep(wait)
        _last_request = time.monotonic()


def search_papers(query: str, limit: int = 5) -> list[WebPaper]:
    """按相关性检索 arXiv 论文。失败抛 WebSearchError。"""
    params = urllib.parse.urlencode(
        {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": limit,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
    )
    _throttle()
    try:
        with urllib.request.urlopen(f"{ARXIV_API}?{params}", timeout=_TIMEOUT) as resp:
            body = resp.read()
    except Exception as exc:
        raise WebSearchError(f"arXiv 请求失败：{exc}") from exc

    try:
        root = ET.fromstring(body)
    except ET.ParseError as exc:
        raise WebSearchError(f"arXiv 响应解析失败：{exc}") from exc

    papers: list[WebPaper] = []
    for entry in root.findall(f"{_ATOM}entry"):
        title = " ".join((entry.findtext(f"{_ATOM}title") or "").split())
        abstract = " ".join((entry.findtext(f"{_ATOM}summary") or "").split())
        authors = [
            name
            for name in ((a.findtext(f"{_ATOM}name") or "").strip() for a in entry.findall(f"{_ATOM}author"))
            if name
        ]
        published = entry.findtext(f"{_ATOM}published") or ""
        year = int(published[:4]) if len(published) >= 4 else None
        pdf_url = None
        alt_url = None
        for link in entry.findall(f"{_ATOM}link"):
            rel = link.get("rel")
            href = link.get("href") or ""
            if rel == "related" and link.get("title") == "pdf":
                pdf_url = href
            elif rel == "alternate":
                alt_url = href
        url = alt_url or entry.findtext(f"{_ATOM}id") or ""
        papers.append(
            WebPaper(title=title, authors=authors, year=year, abstract=abstract, url=url, pdf_url=pdf_url)
        )
    return papers
