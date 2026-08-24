"""arXiv Atom adapter preserving the legacy API's rate-limit contract."""

from __future__ import annotations

import threading
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Callable, Optional

from .base import NormalizedSource, SourceProviderError
from .identity import normalize_arxiv_id, normalize_doi

ARXIV_API = "https://export.arxiv.org/api/query"
_ATOM = "{http://www.w3.org/2005/Atom}"
_ARXIV = "{http://arxiv.org/schemas/atom}"
_DEFAULT_TIMEOUT = 20.0
_DEFAULT_MIN_INTERVAL = 3.0
_USER_AGENT = "PRAgent/0.1 (paper research assistant)"


class _RateLimiter:
    def __init__(self, min_interval: float) -> None:
        self.min_interval = min_interval
        self._last_request = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            delay = self._last_request + self.min_interval - now
            if delay > 0:
                time.sleep(delay)
            self._last_request = time.monotonic()


_GLOBAL_LIMITER = _RateLimiter(_DEFAULT_MIN_INTERVAL)


class ArxivAdapter:
    name = "arxiv"

    def __init__(
        self,
        *,
        timeout: float = _DEFAULT_TIMEOUT,
        limiter: Optional[_RateLimiter] = None,
        requester: Optional[Callable[[str, dict[str, str], float], bytes]] = None,
    ) -> None:
        if timeout <= 0:
            raise ValueError("timeout 必须大于 0")
        self.timeout = timeout
        self._limiter = limiter or _GLOBAL_LIMITER
        self._requester = requester or _request_bytes

    def search(self, query: str, *, limit: int = 10) -> list[NormalizedSource]:
        query = str(query).strip()
        if not query:
            raise ValueError("query 不能为空")
        if len(query) > 500:
            raise ValueError("query 不能超过 500 字符")
        _validate_limit(limit)
        params = urllib.parse.urlencode(
            {
                "search_query": f"all:{query}",
                "start": 0,
                "max_results": limit,
                "sortBy": "relevance",
                "sortOrder": "descending",
            }
        )
        return self._fetch(f"{ARXIV_API}?{params}")

    def lookup(self, identifier: str) -> Optional[NormalizedSource]:
        if len(str(identifier)) > 500:
            raise ValueError("identifier 不能超过 500 字符")
        arxiv_id = normalize_arxiv_id(identifier)
        if arxiv_id is None:
            raise ValueError("identifier 不是有效 arXiv ID")
        params = urllib.parse.urlencode({"id_list": arxiv_id, "max_results": 1})
        records = self._fetch(f"{ARXIV_API}?{params}")
        return records[0] if records else None

    def _fetch(self, url: str) -> list[NormalizedSource]:
        self._limiter.wait()
        try:
            body = self._requester(
                url,
                {"User-Agent": _USER_AGENT, "Accept": "application/atom+xml"},
                self.timeout,
            )
        except SourceProviderError:
            raise
        except Exception as exc:
            raise SourceProviderError(
                f"arXiv 请求失败：{exc}",
                provider=self.name,
                code="network_error",
                retryable=True,
            ) from exc
        try:
            return parse_arxiv_feed(body)
        except ET.ParseError as exc:
            raise SourceProviderError(
                f"arXiv 响应解析失败：{exc}",
                provider=self.name,
                code="invalid_response",
            ) from exc


def parse_arxiv_feed(body: bytes | str) -> list[NormalizedSource]:
    root = ET.fromstring(body)
    retrieved_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    records: list[NormalizedSource] = []
    for entry in root.findall(f"{_ATOM}entry"):
        raw_id = (entry.findtext(f"{_ATOM}id") or "").strip()
        arxiv_id = normalize_arxiv_id(raw_id)
        if arxiv_id is None:
            continue
        title = " ".join((entry.findtext(f"{_ATOM}title") or "").split())
        abstract = " ".join((entry.findtext(f"{_ATOM}summary") or "").split())
        authors = tuple(
            name
            for name in (
                " ".join((author.findtext(f"{_ATOM}name") or "").split())
                for author in entry.findall(f"{_ATOM}author")
            )
            if name
        )
        published = (entry.findtext(f"{_ATOM}published") or "").strip()
        year = _year_from_date(published)
        landing_url: Optional[str] = None
        pdf_url: Optional[str] = None
        for link in entry.findall(f"{_ATOM}link"):
            rel = link.get("rel")
            href = (link.get("href") or "").strip()
            if rel == "related" and link.get("title") == "pdf":
                pdf_url = href or None
            elif rel == "alternate":
                landing_url = href or None
        landing_url = landing_url or raw_id or None
        doi = normalize_doi(entry.findtext(f"{_ARXIV}doi"))
        categories = tuple(
            category.get("term")
            for category in entry.findall(f"{_ATOM}category")
            if category.get("term")
        )
        metadata = {
            "id": raw_id,
            "title": title,
            "authors": list(authors),
            "abstract": abstract,
            "published": published or None,
            "updated": (entry.findtext(f"{_ATOM}updated") or "").strip() or None,
            "doi": doi,
            "categories": list(categories),
            "journal_ref": (entry.findtext(f"{_ARXIV}journal_ref") or "").strip()
            or None,
            "landing_url": landing_url,
            "pdf_url": pdf_url,
        }
        records.append(
            NormalizedSource(
                provider="arxiv",
                provider_record_id=arxiv_id,
                title=title,
                authors=authors,
                year=year,
                abstract=abstract,
                doi=doi,
                arxiv_id=arxiv_id,
                canonical_url=landing_url,
                landing_url=landing_url,
                pdf_url=pdf_url,
                metadata=metadata,
                retrieved_at=retrieved_at,
            )
        )
    return records


def _request_bytes(url: str, headers: dict[str, str], timeout: float) -> bytes:
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def _validate_limit(limit: int) -> None:
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
        raise ValueError("limit 必须是 1–100 的整数")


def _year_from_date(value: str) -> Optional[int]:
    if len(value) >= 4 and value[:4].isdigit():
        year = int(value[:4])
        if 1000 <= year <= 9999:
            return year
    return None
