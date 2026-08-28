"""Deterministic JSON artifact export."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from pragent.research import source_to_csl_json

from .models import EXPORT_SCHEMA_VERSION, ExportEnvelope, FrozenArtifactExport


def export_payload(snapshot: FrozenArtifactExport) -> dict[str, Any]:
    payload = {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "project": asdict(snapshot.project),
        "artifact": asdict(snapshot.artifact),
        "revision": asdict(snapshot.revision),
        "freshness": asdict(snapshot.freshness),
        "citation_style": snapshot.citation_style,
        "sources": [_source_payload(item) for item in snapshot.sources],
        "evidence": [_evidence_payload(item) for item in snapshot.evidence],
    }
    return ExportEnvelope.model_validate(payload).model_dump(mode="json")


def render_json(snapshot: FrozenArtifactExport) -> bytes:
    text = json.dumps(
        export_payload(snapshot),
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        separators=(",", ": "),
    )
    return f"{text}\n".encode("utf-8")


def _source_payload(item) -> dict[str, Any]:
    source = item.source
    public_source = {
        "id": source.id,
        "canonical_key": source.canonical_key,
        "source_kind": source.source_kind,
        "title": source.title,
        "authors": list(source.authors),
        "year": source.year,
        "doi": source.doi,
        "arxiv_id": source.arxiv_id,
        "canonical_url": source.canonical_url,
        "content_sha256": source.content_sha256,
        "indexed_paper_id": source.indexed_paper_id,
        "status": source.status,
        "metadata": source.metadata,
        "snapshot_sha256": source.snapshot_sha256,
        "fetched_at": source.fetched_at,
        "version": source.version,
        "created_at": source.created_at,
        "updated_at": source.updated_at,
    }
    return {
        "source": public_source,
        "csl_json": source_to_csl_json(source),
        "identities": [asdict(identity) for identity in item.identities],
        "provenance": [asdict(record) for record in item.records],
    }


def _evidence_payload(item) -> dict[str, Any]:
    evidence = item.evidence
    return {
        "link": asdict(item.link),
        "snapshot": {
            "id": evidence.id,
            "paper_id": evidence.paper_id,
            "chunk_id": evidence.chunk_id,
            "source_hash": evidence.source_hash,
            "paper_sha256": evidence.paper_sha256,
            "chunk_text_sha256": evidence.chunk_text_sha256,
            "title": evidence.title,
            "authors": list(evidence.authors),
            "year": evidence.year,
            "page": evidence.page,
            "chunk_seq": evidence.chunk_seq,
            "text": evidence.text,
            "annotation": evidence.annotation,
            "pinned_at": evidence.pinned_at,
            "stale": evidence.stale,
            "stale_reason": evidence.stale_reason,
        },
    }
