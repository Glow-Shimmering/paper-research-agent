"""Trafilatura-backed text and metadata extraction; raw HTML is never returned."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import trafilatura
from trafilatura.settings import Extractor, use_config

from pragent.sources.identity import canonicalize_url


class HtmlExtractionError(RuntimeError):
    pass


@dataclass(frozen=True)
class ExtractedWebDocument:
    text: str = field(repr=False)
    title: str = ""
    authors: tuple[str, ...] = ()
    year: Optional[int] = None
    published_at: Optional[str] = None
    language: Optional[str] = None
    canonical_url: str = ""
    text_sha256: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)


def extract_html(
    html: bytes,
    *,
    final_url: str,
    max_tree_size: int = 5_000_000,
    max_text_chars: int = 5_000_000,
) -> ExtractedWebDocument:
    if not html:
        raise HtmlExtractionError("HTML 正文为空")
    if max_tree_size <= 0 or max_text_chars <= 0:
        raise ValueError("HTML extraction 限制必须大于 0")
    canonical_url = canonicalize_url(final_url)
    try:
        config = use_config()
        config.set("DEFAULT", "MAX_TREE_SIZE", str(max_tree_size))
        options = Extractor(
            config=config,
            output_format="python",
            comments=False,
            tables=True,
            images=False,
            links=False,
            dedup=True,
            url=canonical_url,
            with_metadata=True,
        )
        document = trafilatura.bare_extraction(html, options=options)
    except Exception as exc:
        raise HtmlExtractionError(f"Trafilatura 抽取失败：{exc}") from exc
    if document is None:
        raise HtmlExtractionError("Trafilatura 未找到可抽取正文")
    raw_text = getattr(document, "text", None) or getattr(document, "raw_text", None)
    text = _normalize_text(raw_text)
    if not text:
        raise HtmlExtractionError("网页没有可抽取文本正文")
    if len(text) > max_text_chars:
        raise HtmlExtractionError("抽取正文超过字符限制")
    title = " ".join(str(getattr(document, "title", None) or "").split())
    authors = _authors(getattr(document, "author", None))
    published_at = _optional_text(getattr(document, "date", None))
    language = _optional_text(getattr(document, "language", None))
    declared_url = _optional_http_url(getattr(document, "url", None))
    metadata = {
        "title": title or None,
        "authors": list(authors),
        "published_at": published_at,
        "language": language,
        "declared_url": declared_url,
        "hostname": _optional_text(getattr(document, "hostname", None)),
        "description": _optional_text(getattr(document, "description", None)),
        "sitename": _optional_text(getattr(document, "sitename", None)),
        "categories": list(getattr(document, "categories", None) or []),
        "tags": list(getattr(document, "tags", None) or []),
    }
    return ExtractedWebDocument(
        text=text,
        title=title,
        authors=authors,
        year=_year(published_at),
        published_at=published_at,
        language=language,
        canonical_url=canonical_url,
        text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        metadata=metadata,
    )


def _normalize_text(value: Any) -> str:
    lines = []
    for line in str(value or "").replace("\x00", "").splitlines():
        normalized = " ".join(line.split())
        if normalized:
            lines.append(normalized)
    return "\n".join(lines)


def _authors(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    raw_values = value if isinstance(value, (list, tuple)) else str(value).split(";")
    return tuple(
        normalized
        for normalized in (" ".join(str(item).split()) for item in raw_values)
        if normalized
    )


def _optional_text(value: Any) -> Optional[str]:
    text = " ".join(str(value or "").split())
    return text or None


def _optional_http_url(value: Any) -> Optional[str]:
    text = _optional_text(value)
    if text is None:
        return None
    try:
        return canonicalize_url(text)
    except ValueError:
        return None


def _year(value: Optional[str]) -> Optional[int]:
    if value and len(value) >= 4 and value[:4].isdigit():
        year = int(value[:4])
        if 1000 <= year <= 9999:
            return year
    return None
