"""Semantic Scholar Graph API adapter with optional API-key authentication."""

from __future__ import annotations

import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

from .base import NormalizedSource, SourceProviderError
from .http import JsonHttpClient, RateLimiter, ResponseCache
from .identity import normalize_arxiv_id, normalize_doi

_API_BASE = "https://api.semanticscholar.org/graph/v1"
_FIELDS = (
    "paperId,title,authors,year,abstract,externalIds,url,openAccessPdf,"
    "publicationDate,venue,publicationTypes"
)
_USER_AGENT = "PRAgent/0.1 (paper research assistant)"


class SemanticScholarAdapter:
    name = "semantic_scholar"

    def __init__(
        self,
        *,
        api_key: str = "",
        client: Optional[JsonHttpClient] = None,
        cache_directory: Optional[str | Path] = None,
        timeout: float = 20.0,
    ) -> None:
        self.api_key = api_key.strip()
        self.client = client or JsonHttpClient(
            self.name,
            cache=ResponseCache(cache_directory) if cache_directory is not None else None,
            limiter=RateLimiter(1.0),
            timeout=timeout,
        )

    def search(self, query: str, *, limit: int = 10) -> list[NormalizedSource]:
        query = str(query).strip()
        if not query:
            raise ValueError("query 不能为空")
        _validate_limit(limit)
        params = urllib.parse.urlencode(
            {"query": query, "limit": limit, "fields": _FIELDS}
        )
        payload = self.client.get_json(
            f"{_API_BASE}/paper/search?{params}", headers=self._headers()
        )
        if not isinstance(payload, Mapping) or not isinstance(payload.get("data"), list):
            raise SourceProviderError(
                "Semantic Scholar 响应缺少 data 列表",
                provider=self.name,
                code="invalid_response",
            )
        return [
            normalized
            for item in payload["data"]
            if isinstance(item, Mapping)
            for normalized in [normalize_semantic_scholar_record(item)]
            if normalized is not None
        ]

    def lookup(self, identifier: str) -> Optional[NormalizedSource]:
        value = str(identifier).strip()
        if not value:
            raise ValueError("identifier 不能为空")
        doi = normalize_doi(value)
        arxiv_id = normalize_arxiv_id(value)
        if doi:
            provider_identifier = f"DOI:{doi}"
        elif arxiv_id:
            provider_identifier = f"ARXIV:{arxiv_id}"
        else:
            provider_identifier = value
        encoded = urllib.parse.quote(provider_identifier, safe="")
        params = urllib.parse.urlencode({"fields": _FIELDS})
        try:
            payload = self.client.get_json(
                f"{_API_BASE}/paper/{encoded}?{params}", headers=self._headers()
            )
        except SourceProviderError as exc:
            if exc.status_code == 404:
                return None
            raise
        if not isinstance(payload, Mapping):
            raise SourceProviderError(
                "Semantic Scholar lookup 响应不是对象",
                provider=self.name,
                code="invalid_response",
            )
        return normalize_semantic_scholar_record(payload)

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/json",
            "User-Agent": _USER_AGENT,
        }
        if self.api_key:
            headers["x-api-key"] = self.api_key
        return headers


def normalize_semantic_scholar_record(
    item: Mapping[str, Any],
) -> Optional[NormalizedSource]:
    paper_id = str(item.get("paperId") or "").strip()
    if not paper_id:
        return None
    external = item.get("externalIds")
    external = external if isinstance(external, Mapping) else {}
    doi = normalize_doi(external.get("DOI") or external.get("doi"))
    arxiv_id = normalize_arxiv_id(
        external.get("ArXiv") or external.get("ARXIV") or external.get("arxiv")
    )
    authors = tuple(
        name
        for author in (item.get("authors") or [])
        if isinstance(author, Mapping)
        for name in [" ".join(str(author.get("name") or "").split())]
        if name
    )
    open_access = item.get("openAccessPdf")
    pdf_url = (
        str(open_access.get("url") or "").strip() or None
        if isinstance(open_access, Mapping)
        else None
    )
    landing_url = str(item.get("url") or "").strip() or None
    title = " ".join(str(item.get("title") or "").split())
    abstract = " ".join(str(item.get("abstract") or "").split())
    year = _year(item.get("year"), item.get("publicationDate"))
    retrieved_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return NormalizedSource(
        provider="semantic_scholar",
        provider_record_id=paper_id,
        title=title,
        authors=authors,
        year=year,
        abstract=abstract,
        doi=doi,
        arxiv_id=arxiv_id,
        canonical_url=landing_url,
        landing_url=landing_url,
        pdf_url=pdf_url,
        metadata=dict(item),
        retrieved_at=retrieved_at,
    )


def _year(raw_year: Any, publication_date: Any) -> Optional[int]:
    if isinstance(raw_year, int) and not isinstance(raw_year, bool) and 1000 <= raw_year <= 9999:
        return raw_year
    text = str(publication_date or "")
    if len(text) >= 4 and text[:4].isdigit():
        value = int(text[:4])
        return value if 1000 <= value <= 9999 else None
    return None


def _validate_limit(limit: int) -> None:
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
        raise ValueError("limit 必须是 1–100 的整数")
