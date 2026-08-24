"""Unified academic/web source provider contracts and adapters."""

from .arxiv import ArxivAdapter
from .base import (
    MergedSource,
    NormalizedSource,
    ProviderProvenance,
    SourceProvider,
    SourceProviderError,
)
from .crossref import CrossrefAdapter
from .discovery import DiscoveryBatch, DiscoveryItem, DiscoveryService, ProviderFailure
from .identity import (
    canonicalize_url,
    deduplicate_sources,
    normalize_arxiv_id,
    normalize_content_sha256,
    normalize_doi,
    source_identities,
)
from .semantic_scholar import SemanticScholarAdapter

__all__ = [
    "ArxivAdapter",
    "MergedSource",
    "NormalizedSource",
    "ProviderProvenance",
    "ProviderFailure",
    "DiscoveryBatch",
    "DiscoveryItem",
    "DiscoveryService",
    "CrossrefAdapter",
    "SemanticScholarAdapter",
    "SourceProvider",
    "SourceProviderError",
    "canonicalize_url",
    "deduplicate_sources",
    "normalize_arxiv_id",
    "normalize_content_sha256",
    "normalize_doi",
    "source_identities",
]
