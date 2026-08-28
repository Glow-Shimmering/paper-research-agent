"""Deterministic UTF-8 CSV exports for sources and comparison matrices."""

from __future__ import annotations

import csv
import io
import json

from .common import parse_artifact_content
from .models import FrozenArtifactExport, RenderedExport


def render_csv(snapshot: FrozenArtifactExport) -> tuple[RenderedExport, ...]:
    files = [
        RenderedExport(
            "sources.csv",
            "text/csv; charset=utf-8",
            _sources_csv(snapshot),
        )
    ]
    if snapshot.artifact.artifact_type == "comparison":
        files.append(
            RenderedExport(
                "comparison.csv",
                "text/csv; charset=utf-8",
                _comparison_csv(snapshot),
            )
        )
    return tuple(files)


def _sources_csv(snapshot: FrozenArtifactExport) -> bytes:
    rows = []
    for item in snapshot.sources:
        source = item.source
        rows.append(
            (
                source.id,
                source.source_kind,
                source.title,
                " | ".join(source.authors),
                "" if source.year is None else str(source.year),
                source.doi or "",
                source.arxiv_id or "",
                source.canonical_url or "",
                source.status,
                " | ".join(record.provider for record in item.records),
                json.dumps(source.metadata, ensure_ascii=False, sort_keys=True),
            )
        )
    return _write_csv(
        (
            "source_id",
            "source_kind",
            "title",
            "authors",
            "year",
            "doi",
            "arxiv_id",
            "canonical_url",
            "status",
            "providers",
            "metadata_json",
        ),
        rows,
    )


def _comparison_csv(snapshot: FrozenArtifactExport) -> bytes:
    matrix = parse_artifact_content(snapshot)
    sources = snapshot.source_by_id()
    dimensions = {item.key: item for item in matrix.dimensions}
    rows = []
    for cell in matrix.cells:
        rows.append(
            (
                cell.dimension_key,
                dimensions[cell.dimension_key].label,
                cell.source_id,
                sources[cell.source_id].title,
                cell.summary,
                "true" if cell.insufficient_evidence else "false",
                " | ".join(ref.evidence_id for ref in cell.evidence_refs),
            )
        )
    return _write_csv(
        (
            "dimension_key",
            "dimension_label",
            "source_id",
            "source_title",
            "summary",
            "insufficient_evidence",
            "evidence_ids",
        ),
        rows,
    )


def _write_csv(header, rows) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")
