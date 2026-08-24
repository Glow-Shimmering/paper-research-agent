"""Deterministic source identity normalization and transitive deduplication."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import posixpath
import re
from dataclasses import replace
from typing import Iterable, Optional
from urllib.parse import parse_qsl, quote, unquote, urlencode, urlsplit, urlunsplit

from .base import MergedSource, NormalizedSource

_DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$", re.IGNORECASE)
_ARXIV_NEW_RE = re.compile(r"^(\d{4}\.\d{4,5})(?:v\d+)?$", re.IGNORECASE)
_ARXIV_OLD_RE = re.compile(
    r"^([a-z][a-z0-9.\-]+(?:/[a-z0-9.\-]+)?/\d{7})(?:v\d+)?$",
    re.IGNORECASE,
)
_PERCENT_ESCAPE_RE = re.compile(r"%[0-9a-fA-F]{2}")
_UNRESERVED = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~"
)
_TRACKING_QUERY_KEYS = frozenset(
    {
        "fbclid",
        "gclid",
        "mc_cid",
        "mc_eid",
        "ref",
        "ref_src",
    }
)
_IDENTITY_PRIORITY = {"doi": 0, "arxiv": 1, "url": 2, "content_sha256": 3}
_PROVIDER_PRIORITY = {
    "crossref": 0,
    "semantic_scholar": 1,
    "arxiv": 2,
    "web": 3,
    "local": 4,
}


def normalize_doi(value: object) -> Optional[str]:
    """Return lowercase bare DOI, or ``None`` for malformed input."""

    if value is None:
        return None
    text = unquote(str(value)).strip()
    lowered = text.lower()
    for prefix in ("doi:", "https://doi.org/", "http://doi.org/", "https://dx.doi.org/", "http://dx.doi.org/"):
        if lowered.startswith(prefix):
            text = text[len(prefix) :].strip()
            break
    text = text.strip().rstrip(".,;)").lower()
    if not _DOI_RE.fullmatch(text) or any(character.isspace() for character in text):
        return None
    return text


def normalize_arxiv_id(value: object) -> Optional[str]:
    """Return versionless modern or legacy arXiv identifier."""

    if value is None:
        return None
    text = unquote(str(value)).strip()
    lowered = text.lower()
    if lowered.startswith("arxiv:"):
        text = text[6:].strip()
    else:
        parsed = urlsplit(text)
        if parsed.scheme and (parsed.hostname or "").lower() in {
            "arxiv.org",
            "www.arxiv.org",
            "export.arxiv.org",
        }:
            path = parsed.path.strip("/")
            for prefix in ("abs/", "pdf/"):
                if path.lower().startswith(prefix):
                    path = path[len(prefix) :]
                    break
            if path.lower().endswith(".pdf"):
                path = path[:-4]
            text = path
    text = text.strip().lower()
    match = _ARXIV_NEW_RE.fullmatch(text) or _ARXIV_OLD_RE.fullmatch(text)
    return match.group(1).lower() if match else None


def normalize_content_sha256(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        return None
    return text


def canonicalize_url(value: object) -> str:
    """Normalize an HTTP(S) identity URL without merging encoded reserved bytes."""

    text = str(value).strip()
    parsed = urlsplit(text)
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"}:
        raise ValueError("canonical URL 只允许 http/https")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("canonical URL 不允许 credentials")
    host = parsed.hostname
    if not host:
        raise ValueError("canonical URL 缺少 host")
    host = _normalize_host(host)
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("canonical URL port 无效") from exc
    default_port = (scheme == "http" and port == 80) or (
        scheme == "https" and port == 443
    )
    host_display = f"[{host}]" if ":" in host else host
    netloc = host_display if port is None or default_port else f"{host_display}:{port}"

    path = quote(
        _normalize_percent_escapes(parsed.path or "/"),
        safe="/:@-._~!$&'()*+,;=%",
    )
    normalized_path = posixpath.normpath(path)
    if not normalized_path.startswith("/"):
        normalized_path = f"/{normalized_path}"
    if normalized_path != "/" and path.endswith("/"):
        normalized_path += "/"
    if normalized_path != "/":
        normalized_path = normalized_path.rstrip("/")

    query_items = []
    for key, item_value in parse_qsl(parsed.query, keep_blank_values=True):
        lowered = key.lower()
        if lowered.startswith("utm_") or lowered in _TRACKING_QUERY_KEYS:
            continue
        query_items.append((key, item_value))
    query_items.sort()
    query = urlencode(query_items, doseq=True, quote_via=quote)
    return urlunsplit((scheme, netloc, normalized_path, query, ""))


def source_identities(source: NormalizedSource) -> tuple[tuple[str, str], ...]:
    """Return all valid deterministic identities in merge priority order."""

    values: list[tuple[str, Optional[str]]] = [
        ("doi", normalize_doi(source.doi)),
        ("arxiv", normalize_arxiv_id(source.arxiv_id)),
        ("url", _optional_canonical_url(source.canonical_url or source.landing_url)),
        ("content_sha256", normalize_content_sha256(source.content_sha256)),
    ]
    return tuple((kind, value) for kind, value in values if value is not None)


def canonical_key(source: NormalizedSource) -> str:
    identities = source_identities(source)
    if identities:
        kind, value = identities[0]
        return f"{kind}:{value}"
    fallback = f"{source.provider.strip().lower()}\0{source.provider_record_id.strip()}"
    return f"record_sha256:{hashlib.sha256(fallback.encode('utf-8')).hexdigest()}"


def deduplicate_sources(sources: Iterable[NormalizedSource]) -> tuple[MergedSource, ...]:
    """Merge records sharing any identity, including transitive bridges.

    Titles and author strings are deliberately never used as identity. Input order does
    not affect grouping, canonical metadata selection, or output order.
    """

    normalized = tuple(_normalized_copy(source) for source in sources)
    if not normalized:
        return ()
    parents = list(range(len(normalized)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return
        if left_root < right_root:
            parents[right_root] = left_root
        else:
            parents[left_root] = right_root

    identity_owner: dict[tuple[str, str], int] = {}
    fallback_owner: dict[tuple[str, str], int] = {}
    for index, source in enumerate(normalized):
        identities = source_identities(source)
        for identity in identities:
            previous = identity_owner.setdefault(identity, index)
            union(index, previous)
        if not identities:
            record_key = (source.provider.lower(), source.provider_record_id)
            previous = fallback_owner.setdefault(record_key, index)
            union(index, previous)

    groups: dict[int, list[NormalizedSource]] = {}
    for index, source in enumerate(normalized):
        groups.setdefault(find(index), []).append(source)
    merged = [_merge_group(records) for records in groups.values()]
    return tuple(sorted(merged, key=lambda item: item.canonical_key))


def _merge_group(records: list[NormalizedSource]) -> MergedSource:
    by_provider_record: dict[tuple[str, str], NormalizedSource] = {}
    for record in records:
        key = (record.provider, record.provider_record_id)
        current = by_provider_record.get(key)
        if current is None or _record_variant_key(record) > _record_variant_key(current):
            by_provider_record[key] = record
    ordered = sorted(by_provider_record.values(), key=_record_sort_key)
    all_identities = sorted(
        {identity for record in ordered for identity in source_identities(record)},
        key=lambda item: (_IDENTITY_PRIORITY[item[0]], item[1]),
    )
    if all_identities:
        key_kind, key_value = all_identities[0]
        key = f"{key_kind}:{key_value}"
    else:
        key = canonical_key(ordered[0])

    def first_text(field: str) -> Optional[str]:
        for record in ordered:
            value = getattr(record, field)
            if value:
                return str(value)
        return None

    def first_authors() -> tuple[str, ...]:
        return next((record.authors for record in ordered if record.authors), ())

    def first_year() -> Optional[int]:
        return next((record.year for record in ordered if record.year is not None), None)

    identity_map = {kind: value for kind, value in all_identities}
    representative = ordered[0]
    providers = tuple(sorted({record.provider for record in ordered}))
    combined_metadata = dict(representative.metadata)
    combined_metadata["providers"] = list(providers)
    canonical = NormalizedSource(
        provider=representative.provider,
        provider_record_id=representative.provider_record_id,
        source_kind=(
            "web" if all(record.source_kind == "web" for record in ordered) else "paper"
        ),
        title=first_text("title") or "",
        authors=first_authors(),
        year=first_year(),
        abstract=first_text("abstract") or "",
        doi=identity_map.get("doi"),
        arxiv_id=identity_map.get("arxiv"),
        canonical_url=identity_map.get("url"),
        content_sha256=identity_map.get("content_sha256"),
        landing_url=first_text("landing_url"),
        pdf_url=first_text("pdf_url"),
        metadata=combined_metadata,
        retrieved_at=first_text("retrieved_at"),
    )
    provenance_by_key = {
        (record.provider, record.provider_record_id): record.provenance
        for record in ordered
    }
    provenance = tuple(
        provenance_by_key[key_value]
        for key_value in sorted(provenance_by_key)
    )
    return MergedSource(
        canonical_key=key,
        source=canonical,
        identities=tuple(all_identities),
        provenance=provenance,
        duplicate_count=max(0, len(provenance) - 1),
    )


def _normalized_copy(source: NormalizedSource) -> NormalizedSource:
    if not source.provider.strip() or not source.provider_record_id.strip():
        raise ValueError("provider 和 provider_record_id 不能为空")
    if source.source_kind not in {"paper", "web"}:
        raise ValueError("source_kind 必须是 paper 或 web")
    authors = tuple(
        text
        for text in (" ".join(str(author).split()) for author in source.authors)
        if text
    )
    return replace(
        source,
        provider=source.provider.strip().lower(),
        provider_record_id=source.provider_record_id.strip(),
        title=" ".join(source.title.split()),
        authors=authors,
        doi=normalize_doi(source.doi),
        arxiv_id=normalize_arxiv_id(source.arxiv_id),
        canonical_url=_optional_canonical_url(source.canonical_url or source.landing_url),
        content_sha256=normalize_content_sha256(source.content_sha256),
    )


def _record_sort_key(source: NormalizedSource) -> tuple[int, str, str]:
    return (
        _PROVIDER_PRIORITY.get(source.provider, 100),
        source.provider,
        source.provider_record_id,
    )


def _record_variant_key(source: NormalizedSource) -> tuple[str, str]:
    payload = json.dumps(
        {
            "title": source.title,
            "authors": source.authors,
            "year": source.year,
            "abstract": source.abstract,
            "doi": source.doi,
            "arxiv_id": source.arxiv_id,
            "canonical_url": source.canonical_url,
            "content_sha256": source.content_sha256,
            "landing_url": source.landing_url,
            "pdf_url": source.pdf_url,
            "metadata": source.metadata,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return source.retrieved_at or "", hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _optional_canonical_url(value: object) -> Optional[str]:
    if value is None or not str(value).strip():
        return None
    try:
        return canonicalize_url(value)
    except ValueError:
        return None


def _normalize_host(host: str) -> str:
    lowered = host.rstrip(".").lower()
    try:
        return ipaddress.ip_address(lowered).compressed
    except ValueError:
        try:
            return lowered.encode("idna").decode("ascii")
        except UnicodeError as exc:
            raise ValueError("canonical URL host 无效") from exc


def _normalize_percent_escapes(path: str) -> str:
    def replace_escape(match: re.Match[str]) -> str:
        character = chr(int(match.group(0)[1:], 16))
        return character if character in _UNRESERVED else match.group(0).upper()

    return _PERCENT_ESCAPE_RE.sub(replace_escape, path)
