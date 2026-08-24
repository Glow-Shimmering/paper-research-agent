"""Unified academic/web source provider contracts and adapters."""

from .base import (
    MergedSource,
    NormalizedSource,
    ProviderProvenance,
    SourceProvider,
    SourceProviderError,
)
from .identity import (
    canonicalize_url,
    deduplicate_sources,
    normalize_arxiv_id,
    normalize_content_sha256,
    normalize_doi,
    source_identities,
)

__all__ = [
    "MergedSource",
    "NormalizedSource",
    "ProviderProvenance",
    "SourceProvider",
    "SourceProviderError",
    "canonicalize_url",
    "deduplicate_sources",
    "normalize_arxiv_id",
    "normalize_content_sha256",
    "normalize_doi",
    "source_identities",
]
