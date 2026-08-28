"""Shared schema parsing and citation ordering for export renderers."""

from __future__ import annotations

from typing import Any, Iterable

from pragent.research import (
    ComparisonMatrix,
    DeepReadCard,
    ReviewOutline,
    ReviewSectionDraft,
    render_citation_document,
)

from .models import FrozenArtifactExport


class ExportError(RuntimeError):
    pass


class ExportSnapshotConflict(ExportError):
    pass


class UnsupportedArtifactError(ExportError):
    pass


def parse_artifact_content(snapshot: FrozenArtifactExport) -> Any:
    return parse_content(
        snapshot.artifact.artifact_type, snapshot.revision.content
    )


def parse_content(artifact_type: str, raw_content: Any) -> Any:
    parsers = {
        "deep_read": DeepReadCard,
        "comparison": ComparisonMatrix,
        "review_outline": ReviewOutline,
        "review_section": ReviewSectionDraft,
    }
    try:
        parser = parsers[artifact_type]
    except KeyError as exc:
        raise UnsupportedArtifactError(
            f"暂不支持导出 artifact 类型：{artifact_type}"
        ) from exc
    try:
        return parser.model_validate(raw_content)
    except Exception as exc:
        raise ExportError("artifact revision content 不符合其 schema") from exc


def artifact_source_ids(
    artifact_type: str, content: Any, source_id: str | None
) -> tuple[str, ...]:
    if artifact_type == "deep_read":
        if not source_id:
            raise ExportError("deep_read artifact 缺少 source_id")
        return (source_id,)
    if artifact_type in {"comparison", "review_outline"}:
        return tuple(content.source_ids)
    if artifact_type == "review_section":
        return _ordered_unique(
            token.source_id
            for claim in content.claims
            for token in claim.citation_tokens
        )
    raise UnsupportedArtifactError(f"暂不支持导出 artifact 类型：{artifact_type}")


def citation_clusters(
    snapshot: FrozenArtifactExport, content: Any
) -> tuple[tuple[str, ...], ...]:
    artifact_type = snapshot.artifact.artifact_type
    if artifact_type == "deep_read":
        return tuple(
            (snapshot.artifact.source_id,) if field.evidence_refs else ()
            for _name, field in content.ordered_fields()
        )
    if artifact_type == "comparison":
        return tuple(
            (cell.source_id,) if cell.evidence_refs else () for cell in content.cells
        )
    if artifact_type == "review_outline":
        return tuple(
            _ordered_unique(ref.source_id for ref in claim.evidence_refs)
            for section in content.sections
            for claim in section.planned_claims
        )
    if artifact_type == "review_section":
        return tuple(
            _ordered_unique(ref.source_id for ref in claim.citation_tokens)
            for claim in content.claims
        )
    raise UnsupportedArtifactError(f"暂不支持导出 artifact 类型：{artifact_type}")


def citation_output(snapshot: FrozenArtifactExport, content: Any):
    return render_citation_document(
        (item.source for item in snapshot.sources),
        citation_clusters(snapshot, content),
        snapshot.citation_style,
    )


def _ordered_unique(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value for value in values if value))
