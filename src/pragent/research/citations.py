"""CSL-JSON normalization and bundled CSL style rendering."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime
from importlib import resources
from typing import Any, Iterable, Mapping

from citeproc import (
    Citation,
    CitationItem,
    CitationStylesBibliography,
    CitationStylesStyle,
    formatter,
)
from citeproc.source.json import CiteProcJSON

from pragent.models import ResearchSource


class CitationStyleError(ValueError):
    pass


class CitationRenderError(RuntimeError):
    pass


@dataclass(frozen=True)
class CitationDocument:
    citations: tuple[str, ...]
    bibliography: tuple[str, ...]


@dataclass(frozen=True)
class CitationStyleSpec:
    key: str
    label: str
    filename: str
    csl_id: str
    license_name: str = "CC BY-SA 3.0"
    license_url: str = "https://creativecommons.org/licenses/by-sa/3.0/"


STYLE_REGISTRY: tuple[CitationStyleSpec, ...] = (
    CitationStyleSpec(
        "gb-t-7714-2015-numeric",
        "GB/T 7714-2015（顺序编码）",
        "gb-t-7714-2015-numeric.csl",
        "http://www.zotero.org/styles/china-national-standard-gb-t-7714-2015-numeric",
    ),
    CitationStyleSpec(
        "apa-7",
        "APA 7",
        "apa-7.csl",
        "http://www.zotero.org/styles/apa",
    ),
    CitationStyleSpec(
        "ieee",
        "IEEE",
        "ieee.csl",
        "http://www.zotero.org/styles/ieee",
    ),
    CitationStyleSpec(
        "chicago-author-date",
        "Chicago author-date",
        "chicago-author-date.csl",
        "http://www.zotero.org/styles/chicago-author-date",
    ),
    CitationStyleSpec(
        "mla",
        "MLA 9",
        "mla.csl",
        "http://www.zotero.org/styles/modern-language-association",
    ),
)
_STYLE_BY_KEY = {item.key: item for item in STYLE_REGISTRY}
_TYPE_MAP = {
    "journal-article": "article-journal",
    "proceedings-article": "paper-conference",
    "book-chapter": "chapter",
    "book": "book",
    "report": "report",
    "dissertation": "thesis",
    "posted-content": "article",
}


def get_citation_style(style_key: str) -> CitationStyleSpec:
    try:
        return _STYLE_BY_KEY[str(style_key).strip()]
    except KeyError as exc:
        raise CitationStyleError(f"不支持的引用样式：{style_key}") from exc


def source_to_csl_json(source: ResearchSource) -> dict[str, Any]:
    """Normalize the canonical source view without inventing missing metadata."""

    metadata = source.metadata if isinstance(source.metadata, Mapping) else {}
    item: dict[str, Any] = {
        "id": source.id,
        "type": _source_type(source, metadata),
        "title": source.title or source.canonical_key,
    }
    authors = [_author_to_csl(item) for item in source.authors if str(item).strip()]
    if authors:
        item["author"] = authors
    year = source.year or _metadata_year(metadata)
    if year is not None:
        item["issued"] = {"date-parts": [[year]]}
    if source.doi:
        item["DOI"] = source.doi
    url = source.canonical_url
    if not url and source.arxiv_id:
        url = f"https://arxiv.org/abs/{source.arxiv_id}"
    if url:
        item["URL"] = url
    if source.source_kind == "web" and source.fetched_at:
        accessed = _iso_date_parts(source.fetched_at)
        if accessed:
            item["accessed"] = {"date-parts": [accessed]}
    for csl_key, metadata_keys in (
        ("container-title", ("container-title", "journal", "venue")),
        ("volume", ("volume",)),
        ("issue", ("issue",)),
        ("page", ("page", "pages")),
        ("publisher", ("publisher",)),
        ("publisher-place", ("publisher-place", "publisher_location")),
        ("language", ("language",)),
    ):
        value = _first_metadata_text(metadata, metadata_keys)
        if value:
            item[csl_key] = value
    return item


def render_bibliography(
    sources: Iterable[ResearchSource], style_key: str
) -> tuple[str, ...]:
    source_list = tuple(sources)
    if not source_list:
        return ()
    ids = [item.id for item in source_list]
    if len(ids) != len(set(ids)):
        raise CitationRenderError("参考文献来源 ID 不能重复")
    spec = get_citation_style(style_key)
    try:
        with ExitStack() as stack:
            style_resource = resources.files("pragent").joinpath(
                "styles", spec.filename
            )
            style_path = stack.enter_context(resources.as_file(style_resource))
            style = CitationStylesStyle(str(style_path), validate=True)
            source = CiteProcJSON([source_to_csl_json(item) for item in source_list])
            bibliography = CitationStylesBibliography(
                style, source, formatter.plain
            )
            for source_id in ids:
                citation = Citation([CitationItem(source_id)])
                bibliography.register(citation)
            rendered = bibliography.bibliography()
            return tuple(
                "".join(str(part) for part in entry).strip()
                for entry in rendered
            )
    except CitationStyleError:
        raise
    except Exception as exc:
        raise CitationRenderError(
            f"引用样式 {style_key} 渲染失败：{type(exc).__name__}"
        ) from exc


def render_citation_cluster(
    sources: Iterable[ResearchSource], style_key: str
) -> str:
    source_list = tuple(sources)
    if not source_list:
        return ""
    spec = get_citation_style(style_key)
    try:
        with resources.as_file(
            resources.files("pragent").joinpath("styles", spec.filename)
        ) as style_path:
            bibliography = CitationStylesBibliography(
                CitationStylesStyle(str(style_path), validate=True),
                CiteProcJSON([source_to_csl_json(item) for item in source_list]),
                formatter.plain,
            )
            citation = Citation([CitationItem(item.id) for item in source_list])
            bibliography.register(citation)
            return str(bibliography.cite(citation, lambda _item: None)).strip()
    except Exception as exc:
        raise CitationRenderError(
            f"引用样式 {style_key} 渲染失败：{type(exc).__name__}"
        ) from exc


def render_citation_document(
    sources: Iterable[ResearchSource],
    clusters: Iterable[Iterable[str]],
    style_key: str,
) -> CitationDocument:
    """Render all clusters in one processor context so numeric labels stay aligned."""

    source_list = tuple(sources)
    ids = [item.id for item in source_list]
    if len(ids) != len(set(ids)):
        raise CitationRenderError("参考文献来源 ID 不能重复")
    known_ids = set(ids)
    cluster_ids = tuple(tuple(dict.fromkeys(items)) for items in clusters)
    unknown = sorted(
        {source_id for items in cluster_ids for source_id in items} - known_ids
    )
    if unknown:
        raise CitationRenderError(f"引用包含未知来源：{', '.join(unknown)}")
    spec = get_citation_style(style_key)
    try:
        with resources.as_file(
            resources.files("pragent").joinpath("styles", spec.filename)
        ) as style_path:
            bibliography = CitationStylesBibliography(
                CitationStylesStyle(str(style_path), validate=True),
                CiteProcJSON([source_to_csl_json(item) for item in source_list]),
                formatter.plain,
            )
            citations = [
                Citation([CitationItem(source_id) for source_id in items])
                for items in cluster_ids
            ]
            cited_ids = {source_id for items in cluster_ids for source_id in items}
            for citation in citations:
                bibliography.register(citation)
            for source_id in ids:
                if source_id not in cited_ids:
                    bibliography.register(Citation([CitationItem(source_id)]))
            rendered_citations = tuple(
                str(bibliography.cite(citation, lambda _item: None)).strip()
                if citation.cites
                else ""
                for citation in citations
            )
            rendered_bibliography = tuple(
                "".join(str(part) for part in entry).strip()
                for entry in bibliography.bibliography()
            )
            return CitationDocument(rendered_citations, rendered_bibliography)
    except CitationStyleError:
        raise
    except Exception as exc:
        raise CitationRenderError(
            f"引用样式 {style_key} 渲染失败：{type(exc).__name__}"
        ) from exc


def _source_type(source: ResearchSource, metadata: Mapping[str, Any]) -> str:
    if source.source_kind == "web":
        return "webpage"
    raw_type = _first_metadata_text(metadata, ("type",))
    return _TYPE_MAP.get(raw_type, "article-journal")


def _author_to_csl(raw: Any) -> dict[str, str]:
    text = " ".join(str(raw).split())
    if "," in text:
        family, given = (item.strip() for item in text.split(",", 1))
        return {key: value for key, value in (("family", family), ("given", given)) if value}
    parts = text.split()
    if len(parts) > 1 and all(ord(character) < 128 for character in text):
        return {"family": parts[-1], "given": " ".join(parts[:-1])}
    return {"literal": text}


def _first_metadata_text(
    metadata: Mapping[str, Any], keys: Iterable[str]
) -> str:
    for key in keys:
        value = metadata.get(key)
        if isinstance(value, (list, tuple)):
            value = next((item for item in value if str(item).strip()), "")
        if value is not None and str(value).strip():
            return " ".join(str(value).split())
    return ""


def _metadata_year(metadata: Mapping[str, Any]) -> int | None:
    for key in ("published-print", "published-online", "issued", "published"):
        value = metadata.get(key)
        if isinstance(value, Mapping):
            parts = value.get("date-parts")
            if isinstance(parts, list) and parts and isinstance(parts[0], list) and parts[0]:
                value = parts[0][0]
        try:
            year = int(value)
        except (TypeError, ValueError):
            continue
        if 1000 <= year <= 9999:
            return year
    return None


def _iso_date_parts(value: str) -> list[int]:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return []
    return [parsed.year, parsed.month, parsed.day]
