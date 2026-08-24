"""Explicit URL ingestion: safe fetch → snapshot → extraction → source persistence."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from pragent.models import ResearchSource
from pragent.sources.base import MergedSource, NormalizedSource, ProviderProvenance
from pragent.sources.identity import canonicalize_url, deduplicate_sources

from .html_extract import ExtractedWebDocument, extract_html
from .safe_fetch import SafeFetchResult, SafeFetcher
from .snapshots import SnapshotRef, SnapshotStore


@dataclass(frozen=True)
class WebIngestResult:
    source: ResearchSource
    fetch: SafeFetchResult = field(repr=False)
    snapshot: SnapshotRef
    document: ExtractedWebDocument = field(repr=False)


class WebIngestService:
    def __init__(
        self,
        repository,
        *,
        fetcher: Optional[SafeFetcher] = None,
        snapshots: SnapshotStore,
    ) -> None:
        self.repository = repository
        self.fetcher = fetcher or SafeFetcher()
        self.snapshots = snapshots

    def ingest(self, url: str) -> WebIngestResult:
        requested_url = canonicalize_url(url)
        placeholder = deduplicate_sources(
            [
                NormalizedSource(
                    provider="web",
                    provider_record_id=requested_url,
                    source_kind="web",
                    canonical_url=requested_url,
                    landing_url=requested_url,
                    metadata={"requested_url": requested_url, "state": "discovered"},
                    retrieved_at=_now_iso(),
                )
            ]
        )[0]
        source = self.repository.upsert_merged_source(placeholder)
        source = self.repository.update_source(
            source.id,
            expected_version=source.version,
            status="fetching",
        )
        active_source_id = source.id
        try:
            fetched = self.fetcher.fetch(requested_url)
            snapshot = self.snapshots.save(fetched.body)
            document = extract_html(fetched.body, final_url=fetched.final_url)
            final_bundle = _web_bundle(
                requested_url=requested_url,
                fetched=fetched,
                snapshot=snapshot,
                document=document,
            )
            source = self.repository.upsert_merged_source(final_bundle)
            active_source_id = source.id
            source = self.repository.update_source(
                source.id,
                expected_version=source.version,
                title=document.title or source.title,
                authors=document.authors,
                year=document.year,
                canonical_url=fetched.final_url,
                content_sha256=snapshot.sha256,
                status="ready",
                metadata={**source.metadata, **final_bundle.source.metadata},
                locator={
                    "kind": "web_snapshot",
                    "final_url": fetched.final_url,
                    "snapshot_sha256": snapshot.sha256,
                },
                snapshot_path=snapshot.relative_path,
                snapshot_sha256=snapshot.sha256,
                extracted_text=document.text,
                fetched_at=_now_iso(),
            )
            return WebIngestResult(source, fetched, snapshot, document)
        except Exception as exc:
            self._mark_failed(active_source_id, exc)
            raise

    def _mark_failed(self, source_id: str, error: BaseException) -> None:
        current = self.repository.get_source(source_id)
        if current is None:
            return
        code = getattr(error, "code", error.__class__.__name__.lower())
        metadata = dict(current.metadata)
        metadata["last_error"] = {
            "code": str(code),
            "message": str(error)[:500],
            "at": _now_iso(),
        }
        try:
            self.repository.update_source(
                source_id,
                expected_version=current.version,
                status="failed",
                metadata=metadata,
            )
        except Exception:
            # Preserve the original fetch/extraction failure. A concurrent writer may
            # already have recovered this source, so a stale failure must not win.
            return


def _web_bundle(
    *,
    requested_url: str,
    fetched: SafeFetchResult,
    snapshot: SnapshotRef,
    document: ExtractedWebDocument,
) -> MergedSource:
    final_url = canonicalize_url(fetched.final_url)
    metadata = {
        **document.metadata,
        "requested_url": requested_url,
        "final_url": final_url,
        "content_type": fetched.content_type,
        "redirect_count": len(fetched.redirect_chain),
        "snapshot_sha256": snapshot.sha256,
        "text_sha256": document.text_sha256,
    }
    source = NormalizedSource(
        provider="web",
        provider_record_id=final_url,
        source_kind="web",
        title=document.title,
        authors=document.authors,
        year=document.year,
        canonical_url=final_url,
        content_sha256=snapshot.sha256,
        landing_url=final_url,
        metadata=metadata,
        retrieved_at=_now_iso(),
    )
    identities = [("url", final_url)]
    if requested_url != final_url:
        identities.append(("url", requested_url))
    identities.append(("content_sha256", snapshot.sha256))
    provenance = [
        ProviderProvenance(
            provider="web",
            record_id=final_url,
            record_url=final_url,
            raw_metadata=metadata,
            retrieved_at=source.retrieved_at,
        )
    ]
    if requested_url != final_url:
        provenance.append(
            ProviderProvenance(
                provider="web",
                record_id=requested_url,
                record_url=requested_url,
                raw_metadata={
                    "requested_url": requested_url,
                    "final_url": final_url,
                    "redirected": True,
                },
                retrieved_at=source.retrieved_at,
            )
        )
    return MergedSource(
        canonical_key=f"url:{final_url}",
        source=source,
        identities=tuple(identities),
        provenance=tuple(provenance),
        duplicate_count=max(0, len(provenance) - 1),
    )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")
