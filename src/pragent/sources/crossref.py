"""Crossref REST adapter using polite identification and bounded JSON transport."""

from __future__ import annotations

import html
import re
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

from .base import NormalizedSource, SourceProviderError
from .http import JsonHttpClient, RateLimiter, ResponseCache
from .identity import normalize_doi

_API_BASE = "https://api.crossref.org"
_SELECT_FIELDS = (
    "DOI,title,author,published,published-print,published-online,URL,abstract,"
    "type,resource,link,container-title,language"
)
_TAG_RE = re.compile(r"<[^>]+>")


class CrossrefAdapter:
    name = "crossref"

    def __init__(
        self,
        *,
        contact_email: str = "",
        client: Optional[JsonHttpClient] = None,
        cache_directory: Optional[str | Path] = None,
        timeout: float = 20.0,
    ) -> None:
        self.contact_email = contact_email.strip()
        self.client = client or JsonHttpClient(
            self.name,
            cache=ResponseCache(cache_directory) if cache_directory is not None else None,
            limiter=RateLimiter(0.1),
            timeout=timeout,
        )

    def search(self, query: str, *, limit: int = 10) -> list[NormalizedSource]:
        query = str(query).strip()
        if not query:
            raise ValueError("query 不能为空")
        _validate_limit(limit)
        params = urllib.parse.urlencode(
            {
                "query.bibliographic": query,
                "rows": limit,
                "select": _SELECT_FIELDS,
            }
        )
        payload = self.client.get_json(
            f"{_API_BASE}/works?{params}", headers=self._headers()
        )
        message = payload.get("message") if isinstance(payload, Mapping) else None
        items = message.get("items") if isinstance(message, Mapping) else None
        if not isinstance(items, list):
            raise SourceProviderError(
                "Crossref 响应缺少 message.items 列表",
                provider=self.name,
                code="invalid_response",
            )
        return [
            normalized
            for item in items
            if isinstance(item, Mapping)
            for normalized in [normalize_crossref_record(item)]
            if normalized is not None
        ]

    def lookup(self, identifier: str) -> Optional[NormalizedSource]:
        doi = normalize_doi(identifier)
        if doi is None:
            raise ValueError("identifier 不是有效 DOI")
        encoded = urllib.parse.quote(doi, safe="")
        params = urllib.parse.urlencode({"select": _SELECT_FIELDS})
        try:
            payload = self.client.get_json(
                f"{_API_BASE}/works/{encoded}?{params}", headers=self._headers()
            )
        except SourceProviderError as exc:
            if exc.status_code == 404:
                return None
            raise
        message = payload.get("message") if isinstance(payload, Mapping) else None
        if not isinstance(message, Mapping):
            raise SourceProviderError(
                "Crossref lookup 响应缺少 message 对象",
                provider=self.name,
                code="invalid_response",
            )
        return normalize_crossref_record(message)

    def _headers(self) -> dict[str, str]:
        identity = "PRAgent/0.1"
        if self.contact_email:
            identity += f" (mailto:{self.contact_email})"
        return {"Accept": "application/json", "User-Agent": identity}


def normalize_crossref_record(item: Mapping[str, Any]) -> Optional[NormalizedSource]:
    doi = normalize_doi(item.get("DOI"))
    if doi is None:
        return None
    title_value = item.get("title")
    if isinstance(title_value, list):
        title_value = next((value for value in title_value if value), "")
    title = " ".join(str(title_value or "").split())
    authors = tuple(
        name
        for author in (item.get("author") or [])
        if isinstance(author, Mapping)
        for name in [_author_name(author)]
        if name
    )
    year = _published_year(item)
    landing_url = str(item.get("URL") or f"https://doi.org/{doi}").strip()
    pdf_url = _pdf_url(item.get("link"))
    abstract = _plain_abstract(item.get("abstract"))
    retrieved_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return NormalizedSource(
        provider="crossref",
        provider_record_id=doi,
        title=title,
        authors=authors,
        year=year,
        abstract=abstract,
        doi=doi,
        canonical_url=landing_url,
        landing_url=landing_url,
        pdf_url=pdf_url,
        metadata=dict(item),
        retrieved_at=retrieved_at,
    )


def _author_name(author: Mapping[str, Any]) -> str:
    literal = " ".join(str(author.get("name") or "").split())
    if literal:
        return literal
    given = " ".join(str(author.get("given") or "").split())
    family = " ".join(str(author.get("family") or "").split())
    return " ".join(part for part in (given, family) if part)


def _published_year(item: Mapping[str, Any]) -> Optional[int]:
    for key in ("published-print", "published-online", "published", "issued"):
        value = item.get(key)
        if not isinstance(value, Mapping):
            continue
        parts = value.get("date-parts")
        if (
            isinstance(parts, list)
            and parts
            and isinstance(parts[0], list)
            and parts[0]
            and isinstance(parts[0][0], int)
            and not isinstance(parts[0][0], bool)
            and 1000 <= parts[0][0] <= 9999
        ):
            return parts[0][0]
    return None


def _pdf_url(raw_links: Any) -> Optional[str]:
    if not isinstance(raw_links, list):
        return None
    for link in raw_links:
        if not isinstance(link, Mapping):
            continue
        content_type = str(link.get("content-type") or "").lower()
        url = str(link.get("URL") or "").strip()
        if url and content_type == "application/pdf":
            return url
    return None


def _plain_abstract(value: Any) -> str:
    if value is None:
        return ""
    without_tags = _TAG_RE.sub(" ", str(value))
    return " ".join(html.unescape(without_tags).split())


def _validate_limit(limit: int) -> None:
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
        raise ValueError("limit 必须是 1–100 的整数")
