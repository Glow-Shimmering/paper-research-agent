"""Provider-neutral source records used by discovery and persistence."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Protocol, runtime_checkable


class SourceProviderError(RuntimeError):
    """A bounded provider operation failed without returning partial records."""

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        code: str = "provider_error",
        retryable: bool = False,
        status_code: Optional[int] = None,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.code = code
        self.retryable = retryable
        self.status_code = status_code


@dataclass(frozen=True)
class ProviderProvenance:
    """One provider's lossless-enough record retained beside canonical metadata."""

    provider: str
    record_id: str
    raw_metadata: Mapping[str, Any] = field(repr=False, compare=False)
    record_url: Optional[str] = None
    retrieved_at: Optional[str] = None


@dataclass(frozen=True)
class NormalizedSource:
    """A provider result normalized without guessing identity from title/author text."""

    provider: str
    provider_record_id: str
    source_kind: str = "paper"
    title: str = ""
    authors: tuple[str, ...] = ()
    year: Optional[int] = None
    abstract: str = field(default="", repr=False, compare=False)
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    canonical_url: Optional[str] = None
    content_sha256: Optional[str] = None
    landing_url: Optional[str] = None
    pdf_url: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)
    retrieved_at: Optional[str] = None

    @property
    def provenance(self) -> ProviderProvenance:
        return ProviderProvenance(
            provider=self.provider,
            record_id=self.provider_record_id,
            raw_metadata=self.metadata,
            record_url=self.landing_url or self.canonical_url,
            retrieved_at=self.retrieved_at,
        )


@dataclass(frozen=True)
class MergedSource:
    """Deterministically merged canonical view plus every contributing record."""

    canonical_key: str
    source: NormalizedSource
    identities: tuple[tuple[str, str], ...]
    provenance: tuple[ProviderProvenance, ...]
    duplicate_count: int

    @property
    def providers(self) -> tuple[str, ...]:
        return tuple(sorted({item.provider for item in self.provenance}))


@runtime_checkable
class SourceProvider(Protocol):
    """Bounded discovery adapter; implementations must not require credentials."""

    name: str

    def search(self, query: str, *, limit: int = 10) -> list[NormalizedSource]: ...

    def lookup(self, identifier: str) -> Optional[NormalizedSource]: ...
