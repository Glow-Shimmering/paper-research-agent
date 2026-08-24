"""Deep Read 生成结果的证据验证与原子 artifact 保存。"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from pragent.models import ArtifactRevision, ResearchArtifact
from pragent.storage import RecordVersionConflictError, ResearchRepository

from .deep_read import DeepReadWorkflow
from .schemas import DEEP_READ_FIELD_ORDER, DeepReadCard


@dataclass(frozen=True)
class SavedDeepRead:
    artifact: ResearchArtifact
    revision: ArtifactRevision


class DeepReadArtifactService:
    def __init__(self, repository: ResearchRepository) -> None:
        self.repository = repository

    def ensure_artifact(self, project_id: str, source_id: str) -> ResearchArtifact:
        existing = self.repository.get_source_artifact(
            project_id, source_id, "deep_read"
        )
        if existing is not None:
            return existing
        source = self.repository.get_source(source_id)
        if source is None or source.indexed_paper_id is None:
            raise ValueError("来源尚未完成全文索引")
        try:
            return self.repository.create_artifact(
                project_id,
                "deep_read",
                source_id=source_id,
                title=f"{source.title or '未命名来源'} · 精读卡",
                status="generating",
            )
        except sqlite3.IntegrityError:
            concurrent = self.repository.get_source_artifact(
                project_id, source_id, "deep_read"
            )
            if concurrent is None:
                raise
            return concurrent

    def generate_and_save(
        self,
        project_id: str,
        source_id: str,
        workflow: DeepReadWorkflow,
        *,
        expected_artifact_version: int | None = None,
    ) -> SavedDeepRead:
        artifact = self.ensure_artifact(project_id, source_id)
        if artifact.project_id != project_id or artifact.source_id != source_id:
            raise ValueError("Deep Read artifact scope 不匹配")
        if (
            expected_artifact_version is not None
            and artifact.version != expected_artifact_version
        ):
            raise RecordVersionConflictError("Deep Read artifact 已在排队后发生变化")
        source = self.repository.get_source(source_id)
        if source is None or source.indexed_paper_id is None:
            raise ValueError("来源尚未完成全文索引")
        freshness = self.repository.artifact_freshness(artifact.id)
        fingerprint = freshness.current_fingerprint
        if not fingerprint:
            raise ValueError("无法计算来源 fingerprint")
        draft = workflow.generate(source.indexed_paper_id)
        refs = []
        for field_name, value in draft.card.ordered_fields():
            for ordinal, ref in enumerate(value.evidence_refs):
                refs.append(
                    (ref.evidence_id, f"$.{field_name}", ordinal, ref.quote)
                )
        revision = self.repository.append_validated_deep_read_revision(
            artifact.id,
            draft.card.model_dump(mode="json"),
            expected_artifact_version=artifact.version,
            expected_source_fingerprint=fingerprint,
            created_by="model",
            evidence_refs=refs,
            model=draft.model,
            usage=draft.usage,
            finish_reason=draft.finish_reason,
            prompt_version=draft.prompt_version,
            schema_version=draft.schema_version,
        )
        updated = self.repository.get_artifact(artifact.id)
        if updated is None:  # pragma: no cover - 同事务提交后仅数据库损坏时发生
            raise RuntimeError("Deep Read artifact 保存后无法读取")
        return SavedDeepRead(updated, revision)

    def edit_field(
        self,
        project_id: str,
        artifact_id: str,
        field_name: str,
        text: str,
        *,
        expected_artifact_version: int,
    ) -> SavedDeepRead:
        artifact, revision, card = self._current_card(
            project_id, artifact_id, expected_artifact_version
        )
        if field_name not in DEEP_READ_FIELD_ORDER:
            raise ValueError("未知精读字段")
        payload = card.model_dump(mode="json")
        payload[field_name]["text"] = str(text).strip()
        updated_card = DeepReadCard.model_validate(payload)
        links = self.repository.list_artifact_evidence(revision.id)
        new_revision = self.repository.append_artifact_revision(
            artifact.id,
            updated_card.model_dump(mode="json"),
            expected_artifact_version=artifact.version,
            created_by="user",
            evidence_links=[
                (link.evidence_id, link.field_path, link.ordinal) for link in links
            ],
            source_fingerprint=revision.source_fingerprint,
            status="ready",
        )
        updated = self.repository.get_artifact(artifact.id)
        if updated is None:  # pragma: no cover
            raise RuntimeError("Deep Read artifact 编辑后无法读取")
        return SavedDeepRead(updated, new_revision)

    def regenerate_field(
        self,
        project_id: str,
        artifact_id: str,
        field_name: str,
        workflow: DeepReadWorkflow,
        *,
        expected_artifact_version: int,
        base_revision_id: str,
    ) -> SavedDeepRead:
        artifact, revision, card = self._current_card(
            project_id, artifact_id, expected_artifact_version
        )
        if revision.id != base_revision_id:
            raise RecordVersionConflictError("Deep Read base revision 已变化")
        if field_name not in DEEP_READ_FIELD_ORDER:
            raise ValueError("未知精读字段")
        freshness = self.repository.artifact_freshness(artifact.id)
        if freshness.stale or not freshness.current_fingerprint:
            raise ValueError("来源已变化，请先完整重新生成精读卡")
        source = self.repository.get_source(artifact.source_id)
        if source is None or source.indexed_paper_id is None:
            raise ValueError("来源尚未完成全文索引")
        field_draft = workflow.generate_field(source.indexed_paper_id, field_name)
        payload = card.model_dump(mode="json")
        payload[field_name] = field_draft.field.model_dump(mode="json")
        updated_card = DeepReadCard.model_validate(payload)
        refs = []
        for name, value in updated_card.ordered_fields():
            for ordinal, ref in enumerate(value.evidence_refs):
                refs.append((ref.evidence_id, f"$.{name}", ordinal, ref.quote))
        new_revision = self.repository.append_validated_deep_read_revision(
            artifact.id,
            updated_card.model_dump(mode="json"),
            expected_artifact_version=artifact.version,
            expected_source_fingerprint=freshness.current_fingerprint,
            created_by="model",
            evidence_refs=refs,
            model=field_draft.model,
            usage=field_draft.usage,
            finish_reason=field_draft.finish_reason,
            prompt_version=field_draft.prompt_version,
            schema_version=field_draft.schema_version,
        )
        updated = self.repository.get_artifact(artifact.id)
        if updated is None:  # pragma: no cover
            raise RuntimeError("Deep Read artifact 重生成后无法读取")
        return SavedDeepRead(updated, new_revision)

    def _current_card(
        self,
        project_id: str,
        artifact_id: str,
        expected_artifact_version: int,
    ) -> tuple[ResearchArtifact, ArtifactRevision, DeepReadCard]:
        artifact = self.repository.get_artifact(artifact_id)
        if (
            artifact is None
            or artifact.project_id != project_id
            or artifact.artifact_type != "deep_read"
            or artifact.version != expected_artifact_version
        ):
            if artifact is not None and artifact.version != expected_artifact_version:
                raise RecordVersionConflictError("Deep Read artifact 版本冲突")
            raise KeyError("Deep Read artifact 不存在")
        revision = self.repository.get_current_artifact_revision(artifact.id)
        if revision is None:
            raise ValueError("Deep Read 尚无可编辑 revision")
        return artifact, revision, DeepReadCard.model_validate(revision.content)
