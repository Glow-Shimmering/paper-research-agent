"""Safe external document ingestion primitives."""

from .html_extract import ExtractedWebDocument, HtmlExtractionError, extract_html
from .indexing import IndexedSourceResult, index_pdf_source, index_web_source
from .safe_fetch import (
    FetchPolicy,
    SafeFetchError,
    SafeFetchResult,
    SafeFetcher,
    pinned_get,
)
from .snapshots import SnapshotError, SnapshotRef, SnapshotStore
from .web import WebIngestResult, WebIngestService

__all__ = [
    "ExtractedWebDocument",
    "FetchPolicy",
    "HtmlExtractionError",
    "IndexedSourceResult",
    "SafeFetchError",
    "SafeFetchResult",
    "SafeFetcher",
    "SnapshotError",
    "SnapshotRef",
    "SnapshotStore",
    "WebIngestResult",
    "WebIngestService",
    "extract_html",
    "index_pdf_source",
    "index_web_source",
    "pinned_get",
]
