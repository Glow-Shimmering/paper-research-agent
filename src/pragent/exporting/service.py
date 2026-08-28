"""Freeze the current artifact revision, render it, and atomically write exports."""

from __future__ import annotations

import copy
import os
import re
import tempfile
import unicodedata
from dataclasses import replace
from pathlib import Path

from pragent.research import get_citation_style
from pragent.storage import ResearchRepository
from pragent.store import Store

from .common import (
    ExportError,
    ExportSnapshotConflict,
    artifact_source_ids,
    parse_content,
)
from .docx import render_docx
from .json_export import render_json
from .markdown import render_markdown
from .models import (
    ExportedFile,
    FrozenArtifactExport,
    FrozenEvidence,
    FrozenReviewSection,
    FrozenSource,
    RenderedExport,
)
from .tabular import render_csv

_SAFE_SUFFIX = re.compile(r"^[a-z0-9][a-z0-9.-]{0,31}$")
_INVALID_FILENAME = re.compile(r"[<>:\"/\\|?*\x00-\x1f]")
_RESERVED_WINDOWS = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}


class ArtifactExportService:
    def __init__(self, repository: ResearchRepository, store: Store) -> None:
        self.repository = repository
        self.store = store

    def freeze_current(self, artifact_id: str) -> FrozenArtifactExport:
        first = self.repository.get_artifact(artifact_id)
        if first is None:
            raise KeyError(f"研究 artifact 不存在：{artifact_id}")
        revision = self.repository.get_current_artifact_revision(artifact_id)
        if revision is None or revision.revision_number != first.current_revision_number:
            raise ExportSnapshotConflict("artifact current revision 在读取期间变化")
        project = self.repository.get_project(first.project_id)
        if project is None:
            raise ExportError("artifact 所属项目不存在")
        get_citation_style(project.citation_style)
        content = parse_content(first.artifact_type, revision.content)
        source_ids = artifact_source_ids(first.artifact_type, content, first.source_id)
        memberships = self.repository.list_project_sources(
            first.project_id, limit=200
        ).items
        member_ids = {item.source.id for item in memberships}
        if not set(source_ids) <= member_ids:
            raise ExportError("artifact revision 引用了项目外来源")

        sources = []
        for source_id in source_ids:
            source = self.repository.get_source(source_id)
            if source is None:
                raise ExportError(f"artifact 来源不存在：{source_id}")
            sources.append(
                FrozenSource(
                    replace(
                        source,
                        metadata=copy.deepcopy(source.metadata),
                        locator=copy.deepcopy(source.locator),
                    ),
                    self.repository.list_source_identities(source_id),
                    tuple(
                        replace(record, raw_metadata=copy.deepcopy(record.raw_metadata))
                        for record in self.repository.list_source_records(source_id)
                    ),
                )
            )

        review_sections = []
        if first.artifact_type == "review_outline":
            candidates = {}
            for candidate in self.repository.list_artifacts(
                first.project_id, artifact_type="review_section", limit=200
            ).items:
                current = self.repository.get_current_artifact_revision(candidate.id)
                if current is None:
                    continue
                try:
                    draft = parse_content("review_section", current.content)
                except ExportError:
                    continue
                if (
                    draft.outline_artifact_id == first.id
                    and draft.outline_revision_id == revision.id
                    and draft.section_key not in candidates
                ):
                    candidates[draft.section_key] = FrozenReviewSection(
                        candidate,
                        replace(
                            current,
                            content=copy.deepcopy(current.content),
                            usage=copy.deepcopy(current.usage),
                        ),
                        self.repository.artifact_freshness(candidate.id),
                    )
            review_sections = [
                candidates[section.key]
                for section in content.sections
                if section.key in candidates
            ]

        evidence_items = []
        evidence_revisions = (revision, *(item.revision for item in review_sections))
        for evidence_revision in evidence_revisions:
            for link in self.repository.list_artifact_evidence(evidence_revision.id):
                evidence = self.store.get_evidence(link.evidence_id)
                if evidence is None:
                    raise ExportError(f"artifact evidence 不存在：{link.evidence_id}")
                evidence_items.append(FrozenEvidence(link, evidence))

        freshness = self.repository.artifact_freshness(first.id)
        last = self.repository.get_artifact(first.id)
        if (
            last is None
            or last.version != first.version
            or last.current_revision_number != first.current_revision_number
        ):
            raise ExportSnapshotConflict("artifact current revision 在冻结期间变化")
        for item in sources:
            current = self.repository.get_source(item.source.id)
            if current is None or current.version != item.source.version:
                raise ExportSnapshotConflict("artifact 来源在冻结期间变化")
        if self.repository.artifact_freshness(first.id) != freshness:
            raise ExportSnapshotConflict("artifact freshness 在冻结期间变化")
        for item in review_sections:
            current_artifact = self.repository.get_artifact(item.artifact.id)
            current_revision = self.repository.get_current_artifact_revision(
                item.artifact.id
            )
            if (
                current_artifact is None
                or current_revision is None
                or current_artifact.version != item.artifact.version
                or current_revision.id != item.revision.id
                or self.repository.artifact_freshness(item.artifact.id)
                != item.freshness
            ):
                raise ExportSnapshotConflict("review section 在冻结期间变化")

        frozen_revision = replace(
            revision,
            content=copy.deepcopy(revision.content),
            usage=copy.deepcopy(revision.usage),
        )
        return FrozenArtifactExport(
            project=project,
            artifact=first,
            revision=frozen_revision,
            freshness=freshness,
            sources=tuple(sources),
            evidence=tuple(evidence_items),
            review_sections=tuple(review_sections),
        )

    def render(self, snapshot: FrozenArtifactExport, format: str) -> tuple[RenderedExport, ...]:
        normalized = str(format).strip().lower()
        if normalized in {"markdown", "md"}:
            return (
                RenderedExport(
                    "md", "text/markdown; charset=utf-8", render_markdown(snapshot)
                ),
            )
        if normalized == "docx":
            return (
                RenderedExport(
                    "docx",
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    render_docx(snapshot),
                ),
            )
        if normalized == "json":
            return (RenderedExport("json", "application/json", render_json(snapshot)),)
        if normalized == "csv":
            return render_csv(snapshot)
        raise ExportError(f"不支持的导出格式：{format}")

    def export_current(
        self, artifact_id: str, format: str, output_dir: Path | str
    ) -> tuple[ExportedFile, ...]:
        snapshot = self.freeze_current(artifact_id)
        return self.export_snapshot(snapshot, format, output_dir)

    def export_snapshot(
        self,
        snapshot: FrozenArtifactExport,
        format: str,
        output_dir: Path | str,
    ) -> tuple[ExportedFile, ...]:
        rendered = self.render(snapshot, format)
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        base = safe_export_stem(
            snapshot.artifact.title,
            snapshot.artifact.id,
            snapshot.revision.revision_number,
        )
        files = []
        for item in rendered:
            if not _SAFE_SUFFIX.fullmatch(item.suffix):
                raise ExportError("renderer 返回了不安全的文件后缀")
            separator = "-" if "." in item.suffix else "."
            target = directory / f"{base}{separator}{item.suffix}"
            _atomic_write(target, item.data)
            files.append(
                ExportedFile(
                    format=str(format).lower(),
                    media_type=item.media_type,
                    path=target,
                    size=len(item.data),
                )
            )
        return tuple(files)


def safe_export_stem(title: str, artifact_id: str, revision_number: int) -> str:
    normalized = unicodedata.normalize("NFKC", str(title))
    normalized = _INVALID_FILENAME.sub("-", normalized)
    normalized = re.sub(r"\s+", "-", normalized).strip(" .-")
    normalized = re.sub(r"-+", "-", normalized)[:80].rstrip(" .-")
    if not normalized or normalized.upper() in _RESERVED_WINDOWS:
        normalized = "research-artifact"
    safe_id = re.sub(r"[^A-Za-z0-9_-]", "", str(artifact_id))[-12:] or "artifact"
    return f"{normalized}-{safe_id}-r{int(revision_number)}"


def _atomic_write(target: Path, data: bytes) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
