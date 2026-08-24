"""Deep Read 生成结果的证据验证与原子 artifact 保存。"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from pragent.models import ArtifactRevision, ResearchArtifact
from pragent.storage import ResearchRepository

from .deep_read import DeepReadDraft, DeepReadWorkflow


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
    ) -> SavedDeepRead:
        artifact = self.ensure_artifact(project_id, source_id)
        if artifact.project_id != project_id or artifact.source_id != source_id:
            raise ValueError("Deep Read artifact scope 不匹配")
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
