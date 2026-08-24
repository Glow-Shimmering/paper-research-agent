"""Safe external document ingestion primitives."""

from .html_extract import ExtractedWebDocument, HtmlExtractionError, extract_html
from .safe_fetch import FetchPolicy, SafeFetchError, SafeFetchResult, SafeFetcher
from .snapshots import SnapshotError, SnapshotRef, SnapshotStore
from .web import WebIngestResult, WebIngestService

__all__ = [
    "ExtractedWebDocument",
    "FetchPolicy",
    "HtmlExtractionError",
    "SafeFetchError",
    "SafeFetchResult",
    "SafeFetcher",
    "SnapshotError",
    "SnapshotRef",
    "SnapshotStore",
    "WebIngestResult",
    "WebIngestService",
    "extract_html",
]
