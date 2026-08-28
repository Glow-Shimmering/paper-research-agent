"""Evidence-scoped literature review outline workflow."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable, Optional, TypeVar

from pydantic import BaseModel, ValidationError

from pragent.models import ArtifactRevision, ResearchArtifact, ResearchQuestion
from pragent.storage import RecordVersionConflictError, ResearchRepository

from .schemas import (
    REVIEW_OUTLINE_SCHEMA_VERSION,
    REVIEW_SECTION_SCHEMA_VERSION,
    ComparisonMatrix,
    ReviewOutline,
    ReviewOutlinePayload,
    ReviewQuestionSnapshot,
    ReviewSectionDraft,
    ReviewSectionPayload,
)

REVIEW_OUTLINE_PROMPT_VERSION = "review-outline-v1"
REVIEW_SECTION_PROMPT_VERSION = "review-section-v1"


class ReviewOutlineError(RuntimeError):
    pass


class ReviewOutlinePrerequisiteError(ReviewOutlineError):
    pass


class ReviewOutlineSchemaError(ReviewOutlineError):
    pass


class ReviewOutlineBudgetExceeded(ReviewOutlineError):
    pass


@dataclass(frozen=True)
class ReviewOutlineBudget:
    max_sources: int = 20
    max_questions: int = 20
    max_llm_calls: int = 2
    max_context_chars: int = 500_000
    max_reported_tokens: int = 100_000

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} 必须是正整数")
        if self.max_sources > 20 or self.max_questions > 20:
            raise ValueError("Review outline 来源与问题预算不能超过 20")


@dataclass
class ReviewOutlineUsage:
    llm_calls: int = 0
    context_chars: int = 0
    reported_tokens: int = 0
    repair_used: bool = False


@dataclass(frozen=True)
class ReviewOutlineDraft:
    outline: ReviewOutline
    model: str
    usage: dict[str, Any]
    finish_reason: Optional[str]
    prompt_version: str
    schema_version: int


@dataclass(frozen=True)
class SavedReviewOutline:
    artifact: ResearchArtifact
    revision: ArtifactRevision


SchemaT = TypeVar("SchemaT", bound=BaseModel)


class ReviewOutlineWorkflow:
    def __init__(
        self,
        repository: ResearchRepository,
        llm,
        *,
        budget: Optional[ReviewOutlineBudget] = None,
    ) -> None:
        self.repository = repository
        self.llm = llm
        self.budget = budget or ReviewOutlineBudget()
        self.usage = ReviewOutlineUsage()
        self._metadata: list[dict[str, Any]] = []

    def generate(
        self,
        project_id: str,
        question_ids: Iterable[str],
        source_ids: Iterable[str],
        comparison_artifact_id: str,
    ) -> ReviewOutlineDraft:
        if self.llm is None or not hasattr(self.llm, "chat_with_metadata"):
            raise ReviewOutlineError("综述提纲需要可审计 metadata 的 LLM")
        questions = self._validate_questions(project_id, question_ids)
        selected = tuple(str(item).strip() for item in source_ids)
        comparison, comparison_revision, matrix = self._validate_comparison(
            project_id, selected, comparison_artifact_id
        )
        source_titles = {}
        for source_id in selected:
            source = self.repository.get_source(source_id)
            if source is None:  # comparison scope validation already guards this
                raise ReviewOutlinePrerequisiteError("比较来源不存在")
            source_titles[source_id] = source.title or source_id
        system = (
            "你是证据优先的文献综述规划助手。输出严格 JSON。依据研究问题和比较矩阵"
            "规划 2–20 个章节。每个 planned claim 只能引用给定 comparison cell 中"
            "对应 source 的 evidence_id 与逐字 quote；跨论文 claim 的每个 source 都必须"
            "有证据。无法支持时设置 insufficient_evidence=true、evidence_refs=[]，不得"
            "创造来源、证据或改写 quote。"
        )
        user = json.dumps(
            {
                "research_questions": [
                    {
                        "id": item.id,
                        "question": item.question,
                        "version": item.version,
                    }
                    for item in questions
                ],
                "sources": [
                    {"source_id": item, "title": source_titles[item]}
                    for item in selected
                ],
                "comparison": matrix.model_dump(mode="json"),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        payload = self._call_schema(system, user, ReviewOutlinePayload)
        self._validate_output(payload, selected, matrix)
        outline = ReviewOutline(
            title=payload.title,
            research_questions=[
                ReviewQuestionSnapshot(
                    id=item.id,
                    question=item.question,
                    version=item.version,
                )
                for item in questions
            ],
            source_ids=list(selected),
            comparison_artifact_id=comparison.id,
            comparison_revision_id=comparison_revision.id,
            sections=payload.sections,
        )
        return ReviewOutlineDraft(
            outline=outline,
            model=str(getattr(self.llm, "model", "")).strip(),
            usage=self._aggregate_usage(),
            finish_reason=(
                self._metadata[-1].get("finish_reason")
                if self._metadata
                else None
            ),
            prompt_version=REVIEW_OUTLINE_PROMPT_VERSION,
            schema_version=REVIEW_OUTLINE_SCHEMA_VERSION,
        )

    def _validate_questions(
        self, project_id: str, question_ids: Iterable[str]
    ) -> tuple[ResearchQuestion, ...]:
        selected = tuple(str(item).strip() for item in question_ids)
        if not 1 <= len(selected) <= self.budget.max_questions:
            raise ValueError("综述提纲必须选择 1–20 个研究问题")
        if any(not item for item in selected) or len(selected) != len(set(selected)):
            raise ValueError("研究问题不能为空或重复")
        questions = {
            item.id: item for item in self.repository.list_questions(project_id)
        }
        outside = set(selected) - set(questions)
        if outside:
            raise ValueError("研究问题必须全部属于当前项目")
        return tuple(questions[item] for item in selected)

    def _validate_comparison(
        self,
        project_id: str,
        source_ids: tuple[str, ...],
        comparison_artifact_id: str,
    ) -> tuple[ResearchArtifact, ArtifactRevision, ComparisonMatrix]:
        if not 2 <= len(source_ids) <= self.budget.max_sources:
            raise ValueError("综述提纲必须选择 2–20 个来源")
        if any(not item for item in source_ids) or len(source_ids) != len(
            set(source_ids)
        ):
            raise ValueError("综述来源不能为空或重复")
        artifact = self.repository.get_artifact(comparison_artifact_id)
        if (
            artifact is None
            or artifact.project_id != project_id
            or artifact.artifact_type != "comparison"
            or artifact.source_id is not None
            or artifact.status != "ready"
        ):
            raise ReviewOutlinePrerequisiteError("必须选择当前项目已完成的比较矩阵")
        if self.repository.artifact_freshness(artifact.id).stale:
            raise ReviewOutlinePrerequisiteError("比较矩阵已过期，请先重新生成")
        revision = self.repository.get_current_artifact_revision(artifact.id)
        if revision is None:
            raise ReviewOutlinePrerequisiteError("比较矩阵尚无当前 revision")
        matrix = ComparisonMatrix.model_validate(revision.content)
        if tuple(matrix.source_ids) != source_ids:
            raise ReviewOutlinePrerequisiteError(
                "综述来源及顺序必须与当前比较矩阵一致"
            )
        return artifact, revision, matrix

    @staticmethod
    def _validate_output(
        payload: ReviewOutlinePayload,
        selected: tuple[str, ...],
        matrix: ComparisonMatrix,
    ) -> None:
        allowed: dict[str, dict[str, set[str]]] = {
            source_id: {} for source_id in selected
        }
        for cell in matrix.cells:
            for ref in cell.evidence_refs:
                allowed[cell.source_id].setdefault(ref.evidence_id, set()).add(
                    ref.quote
                )
        selected_set = set(selected)
        for section in payload.sections:
            if not set(section.source_ids) <= selected_set:
                raise ReviewOutlineSchemaError("综述 section 引用了未选择来源")
            for claim in section.planned_claims:
                if not set(claim.source_ids) <= selected_set:
                    raise ReviewOutlineSchemaError("综述 claim 引用了未选择来源")
                for ref in claim.evidence_refs:
                    quotes = allowed[ref.source_id].get(ref.evidence_id)
                    if quotes is None or ref.quote not in quotes:
                        raise ReviewOutlineSchemaError(
                            "综述提纲引用了越界 evidence 或改写 quote"
                        )

    def _call_schema(self, system: str, user: str, schema: type[SchemaT]) -> SchemaT:
        response = self._call_llm(system, user)
        try:
            return _parse_schema(response["content"], schema)
        except (ValidationError, ValueError, json.JSONDecodeError) as exc:
            if self.usage.repair_used:
                raise ReviewOutlineSchemaError("综述提纲 schema 验证失败") from exc
            self.usage.repair_used = True
            repaired = self._call_llm(
                "修复下列 JSON，使其严格符合 JSON Schema。只输出 JSON；不得添加输入中不存在的 source、evidence ID 或 quote。",
                json.dumps(
                    {
                        "schema": schema.model_json_schema(),
                        "invalid_output": str(response["content"])[:30000],
                        "validation_error": str(exc)[:3000],
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            )
            try:
                return _parse_schema(repaired["content"], schema)
            except (ValidationError, ValueError, json.JSONDecodeError) as repair_exc:
                raise ReviewOutlineSchemaError(
                    "综述提纲 repair 后仍不符合 schema"
                ) from repair_exc

    def _call_llm(self, system: str, user: str) -> dict[str, Any]:
        self._consume("llm_calls", 1, self.budget.max_llm_calls)
        self._consume(
            "context_chars", len(system) + len(user), self.budget.max_context_chars
        )
        response = self.llm.chat_with_metadata(system, user)
        if not isinstance(response, dict) or not isinstance(
            response.get("content"), str
        ):
            raise ReviewOutlineError("LLM 返回格式无效")
        metadata = response.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}
        self._metadata.append(metadata)
        self._consume(
            "reported_tokens",
            _metadata_total_tokens(metadata),
            self.budget.max_reported_tokens,
        )
        return response

    def _consume(self, name: str, amount: int, maximum: int) -> None:
        current = int(getattr(self.usage, name)) + amount
        if current > maximum:
            raise ReviewOutlineBudgetExceeded(f"综述提纲预算超限：{name}")
        setattr(self.usage, name, current)

    def _aggregate_usage(self) -> dict[str, Any]:
        totals: dict[str, int] = {}
        response_ids = []
        for metadata in self._metadata:
            usage = metadata.get("usage") or {}
            if isinstance(usage, dict):
                for key, value in usage.items():
                    if isinstance(value, int) and not isinstance(value, bool):
                        totals[key] = totals.get(key, 0) + value
            if metadata.get("response_id"):
                response_ids.append(str(metadata["response_id"]))
        return {
            **totals,
            "llm_calls": self.usage.llm_calls,
            "context_chars": self.usage.context_chars,
            "repair_used": self.usage.repair_used,
            "response_ids": response_ids,
        }


class ReviewOutlineArtifactService:
    def __init__(self, repository: ResearchRepository) -> None:
        self.repository = repository

    def generate_and_save(
        self,
        project_id: str,
        question_ids: Iterable[str],
        source_ids: Iterable[str],
        comparison_artifact_id: str,
        workflow: ReviewOutlineWorkflow,
        *,
        title: str = "文献综述提纲",
    ) -> SavedReviewOutline:
        artifact = self.repository.create_artifact(
            project_id,
            "review_outline",
            title=str(title).strip() or "文献综述提纲",
            status="generating",
        )
        draft = workflow.generate(
            project_id, question_ids, source_ids, comparison_artifact_id
        )
        return self._save(
            artifact,
            draft.outline,
            expected_artifact_version=artifact.version,
            created_by="model",
            model=draft.model,
            usage=draft.usage,
            finish_reason=draft.finish_reason,
            prompt_version=draft.prompt_version,
            schema_version=draft.schema_version,
        )

    def edit_section(
        self,
        project_id: str,
        artifact_id: str,
        section_key: str,
        *,
        title: str,
        objective: str,
        expected_artifact_version: int,
    ) -> SavedReviewOutline:
        artifact = self.repository.get_artifact(artifact_id)
        if (
            artifact is None
            or artifact.project_id != project_id
            or artifact.artifact_type != "review_outline"
            or artifact.source_id is not None
        ):
            raise KeyError("综述提纲不存在")
        if artifact.version != expected_artifact_version:
            raise RecordVersionConflictError(f"研究 artifact {artifact_id} 版本冲突")
        if self.repository.artifact_freshness(artifact.id).stale:
            raise ValueError("项目来源已变化，请重新生成综述提纲")
        revision = self.repository.get_current_artifact_revision(artifact.id)
        if revision is None:
            raise ValueError("综述提纲尚无 revision")
        outline = ReviewOutline.model_validate(revision.content).model_copy(deep=True)
        section = next(
            (item for item in outline.sections if item.key == section_key), None
        )
        if section is None:
            raise KeyError("综述提纲 section 不存在")
        section.title = str(title).strip()
        section.objective = str(objective).strip()
        outline = ReviewOutline.model_validate(outline.model_dump(mode="json"))
        return self._save(
            artifact,
            outline,
            expected_artifact_version=expected_artifact_version,
            created_by="user",
            model=None,
            usage={"llm_calls": 0, "operation": "section_edit"},
            finish_reason=None,
            prompt_version="review-outline-user-edit-v1",
            schema_version=REVIEW_OUTLINE_SCHEMA_VERSION,
        )

    def _save(
        self,
        artifact: ResearchArtifact,
        outline: ReviewOutline,
        *,
        expected_artifact_version: int,
        created_by: str,
        model: Optional[str],
        usage: dict[str, Any],
        finish_reason: Optional[str],
        prompt_version: str,
        schema_version: int,
    ) -> SavedReviewOutline:
        refs = []
        for section in outline.sections:
            for claim_index, claim in enumerate(section.planned_claims):
                field_path = f"$.sections.{section.key}.claims.{claim_index}"
                for ordinal, ref in enumerate(claim.evidence_refs):
                    refs.append(
                        (
                            ref.evidence_id,
                            field_path,
                            ordinal,
                            ref.source_id,
                            ref.quote,
                        )
                    )
        revision = self.repository.append_validated_review_outline_revision(
            artifact.id,
            outline.model_dump(mode="json"),
            expected_artifact_version=expected_artifact_version,
            expected_project_fingerprint=self.repository.project_source_fingerprint(
                artifact.project_id
            ),
            question_snapshots=[
                (item.id, item.version, item.question)
                for item in outline.research_questions
            ],
            selected_source_ids=outline.source_ids,
            comparison_artifact_id=outline.comparison_artifact_id,
            comparison_revision_id=outline.comparison_revision_id,
            evidence_refs=refs,
            created_by=created_by,
            model=model,
            usage=usage,
            finish_reason=finish_reason,
            prompt_version=prompt_version,
            schema_version=schema_version,
        )
        updated = self.repository.get_artifact(artifact.id)
        if updated is None:  # pragma: no cover
            raise RuntimeError("Review outline 保存后无法读取")
        return SavedReviewOutline(updated, revision)


@dataclass(frozen=True)
class ReviewSectionBudget:
    max_llm_calls: int = 2
    max_context_chars: int = 250_000
    max_reported_tokens: int = 100_000

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} 必须是正整数")


@dataclass(frozen=True)
class ReviewSectionDraftResult:
    draft: ReviewSectionDraft
    model: str
    usage: dict[str, Any]
    finish_reason: Optional[str]
    prompt_version: str
    schema_version: int


@dataclass(frozen=True)
class SavedReviewSection:
    artifact: ResearchArtifact
    revision: ArtifactRevision


class ReviewSectionWorkflow:
    def __init__(
        self,
        repository: ResearchRepository,
        llm,
        *,
        budget: Optional[ReviewSectionBudget] = None,
    ) -> None:
        self.repository = repository
        self.llm = llm
        self.budget = budget or ReviewSectionBudget()
        self.usage = ReviewOutlineUsage()
        self._metadata: list[dict[str, Any]] = []

    def generate(
        self,
        project_id: str,
        outline_artifact_id: str,
        section_key: str,
    ) -> ReviewSectionDraftResult:
        if self.llm is None or not hasattr(self.llm, "chat_with_metadata"):
            raise ReviewOutlineError("综述章节需要可审计 metadata 的 LLM")
        outline_artifact, outline_revision, outline = self._load_outline(
            project_id, outline_artifact_id
        )
        section = next(
            (item for item in outline.sections if item.key == section_key), None
        )
        if section is None:
            raise KeyError("综述提纲中不存在该 section")
        system = (
            "你是证据优先的文献综述写作助手。只输出严格 JSON claims。每条 claim 的"
            "citation_tokens 只能逐字复用 section planned_claims 中已有的 source_id、"
            "evidence_id、quote，不得扩大证据范围或改写 quote。没有足够证据时必须"
            "设置 insufficient_evidence=true 且 citation_tokens=[]。不要输出参考文献"
            "编号或伪造引文；系统将从结构化 token 确定性渲染引用。"
        )
        user = json.dumps(
            {
                "review_title": outline.title,
                "research_questions": [
                    item.model_dump(mode="json")
                    for item in outline.research_questions
                ],
                "section": section.model_dump(mode="json"),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        payload = self._call_schema(system, user, ReviewSectionPayload)
        self._validate_tokens(payload, section)
        _current_artifact, current_revision, _current_outline = self._load_outline(
            project_id, outline_artifact_id
        )
        if current_revision.id != outline_revision.id:
            raise ReviewOutlinePrerequisiteError(
                "综述提纲在章节生成期间已变化，请重新生成"
            )
        draft = ReviewSectionDraft(
            outline_artifact_id=outline_artifact.id,
            outline_revision_id=outline_revision.id,
            section_key=section.key,
            section_title=section.title,
            claims=payload.claims,
        )
        return ReviewSectionDraftResult(
            draft=draft,
            model=str(getattr(self.llm, "model", "")).strip(),
            usage=self._aggregate_usage(),
            finish_reason=(
                self._metadata[-1].get("finish_reason")
                if self._metadata
                else None
            ),
            prompt_version=REVIEW_SECTION_PROMPT_VERSION,
            schema_version=REVIEW_SECTION_SCHEMA_VERSION,
        )

    def _load_outline(
        self, project_id: str, outline_artifact_id: str
    ) -> tuple[ResearchArtifact, ArtifactRevision, ReviewOutline]:
        artifact = self.repository.get_artifact(outline_artifact_id)
        if (
            artifact is None
            or artifact.project_id != project_id
            or artifact.artifact_type != "review_outline"
            or artifact.source_id is not None
            or artifact.status != "ready"
        ):
            raise ReviewOutlinePrerequisiteError(
                "必须选择当前项目已完成的综述提纲"
            )
        if self.repository.artifact_freshness(artifact.id).stale:
            raise ReviewOutlinePrerequisiteError("综述提纲已过期，请先重新生成")
        revision = self.repository.get_current_artifact_revision(artifact.id)
        if revision is None:
            raise ReviewOutlinePrerequisiteError("综述提纲尚无当前 revision")
        outline = ReviewOutline.model_validate(revision.content)
        questions = {
            item.id: item for item in self.repository.list_questions(project_id)
        }
        if any(
            snapshot.id not in questions
            or questions[snapshot.id].version != snapshot.version
            or questions[snapshot.id].question != snapshot.question
            for snapshot in outline.research_questions
        ):
            raise ReviewOutlinePrerequisiteError(
                "综述提纲绑定的研究问题已变化，请先重新生成"
            )
        comparison = self.repository.get_artifact(outline.comparison_artifact_id)
        comparison_revision = self.repository.get_current_artifact_revision(
            outline.comparison_artifact_id
        )
        if (
            comparison is None
            or comparison.project_id != project_id
            or comparison.artifact_type != "comparison"
            or comparison.status != "ready"
            or comparison_revision is None
            or comparison_revision.id != outline.comparison_revision_id
            or self.repository.artifact_freshness(comparison.id).stale
        ):
            raise ReviewOutlinePrerequisiteError(
                "综述提纲绑定的比较矩阵已变化，请先重新生成"
            )
        return artifact, revision, outline

    @staticmethod
    def _validate_tokens(payload, section) -> None:
        allowed = {
            (ref.source_id, ref.evidence_id, ref.quote)
            for claim in section.planned_claims
            for ref in claim.evidence_refs
        }
        for claim in payload.claims:
            for token in claim.citation_tokens:
                if (token.source_id, token.evidence_id, token.quote) not in allowed:
                    raise ReviewOutlineSchemaError(
                        "综述章节 citation token 超出提纲证据范围"
                    )

    def _call_schema(self, system: str, user: str, schema: type[SchemaT]) -> SchemaT:
        response = self._call_llm(system, user)
        try:
            return _parse_schema(response["content"], schema)
        except (ValidationError, ValueError, json.JSONDecodeError) as exc:
            if self.usage.repair_used:
                raise ReviewOutlineSchemaError("综述章节 schema 验证失败") from exc
            self.usage.repair_used = True
            repaired = self._call_llm(
                "修复下列 JSON，使其严格符合 JSON Schema。只输出 JSON；不得添加输入中不存在的 citation token。",
                json.dumps(
                    {
                        "schema": schema.model_json_schema(),
                        "invalid_output": str(response["content"])[:30000],
                        "validation_error": str(exc)[:3000],
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            )
            try:
                return _parse_schema(repaired["content"], schema)
            except (ValidationError, ValueError, json.JSONDecodeError) as repair_exc:
                raise ReviewOutlineSchemaError(
                    "综述章节 repair 后仍不符合 schema"
                ) from repair_exc

    def _call_llm(self, system: str, user: str) -> dict[str, Any]:
        self._consume("llm_calls", 1, self.budget.max_llm_calls)
        self._consume(
            "context_chars", len(system) + len(user), self.budget.max_context_chars
        )
        response = self.llm.chat_with_metadata(system, user)
        if not isinstance(response, dict) or not isinstance(
            response.get("content"), str
        ):
            raise ReviewOutlineError("LLM 返回格式无效")
        metadata = response.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}
        self._metadata.append(metadata)
        self._consume(
            "reported_tokens",
            _metadata_total_tokens(metadata),
            self.budget.max_reported_tokens,
        )
        return response

    def _consume(self, name: str, amount: int, maximum: int) -> None:
        current = int(getattr(self.usage, name)) + amount
        if current > maximum:
            raise ReviewOutlineBudgetExceeded(f"综述章节预算超限：{name}")
        setattr(self.usage, name, current)

    def _aggregate_usage(self) -> dict[str, Any]:
        totals: dict[str, int] = {}
        response_ids = []
        for metadata in self._metadata:
            usage = metadata.get("usage") or {}
            if isinstance(usage, dict):
                for key, value in usage.items():
                    if isinstance(value, int) and not isinstance(value, bool):
                        totals[key] = totals.get(key, 0) + value
            if metadata.get("response_id"):
                response_ids.append(str(metadata["response_id"]))
        return {
            **totals,
            "llm_calls": self.usage.llm_calls,
            "context_chars": self.usage.context_chars,
            "repair_used": self.usage.repair_used,
            "response_ids": response_ids,
        }


class ReviewSectionArtifactService:
    def __init__(self, repository: ResearchRepository) -> None:
        self.repository = repository

    def generate_and_save(
        self,
        project_id: str,
        outline_artifact_id: str,
        section_key: str,
        workflow: ReviewSectionWorkflow,
    ) -> SavedReviewSection:
        draft = workflow.generate(project_id, outline_artifact_id, section_key)
        artifact = self.repository.create_artifact(
            project_id,
            "review_section",
            title=draft.draft.section_title,
            status="generating",
        )
        return self._save(
            artifact,
            draft.draft,
            expected_artifact_version=artifact.version,
            created_by="model",
            model=draft.model,
            usage=draft.usage,
            finish_reason=draft.finish_reason,
            prompt_version=draft.prompt_version,
            schema_version=draft.schema_version,
        )

    def edit_claim(
        self,
        project_id: str,
        artifact_id: str,
        claim_key: str,
        text: str,
        *,
        expected_artifact_version: int,
        insufficient_evidence: Optional[bool] = None,
    ) -> SavedReviewSection:
        artifact = self.repository.get_artifact(artifact_id)
        if (
            artifact is None
            or artifact.project_id != project_id
            or artifact.artifact_type != "review_section"
            or artifact.source_id is not None
        ):
            raise KeyError("综述章节不存在")
        if artifact.version != expected_artifact_version:
            raise RecordVersionConflictError(f"研究 artifact {artifact_id} 版本冲突")
        if self.repository.artifact_freshness(artifact.id).stale:
            raise ValueError("项目来源已变化，请重新生成综述章节")
        revision = self.repository.get_current_artifact_revision(artifact.id)
        if revision is None:
            raise ValueError("综述章节尚无 revision")
        draft = ReviewSectionDraft.model_validate(revision.content).model_copy(
            deep=True
        )
        claim = next((item for item in draft.claims if item.key == claim_key), None)
        if claim is None:
            raise KeyError("综述章节 claim 不存在")
        claim.text = str(text).strip()
        if insufficient_evidence is True:
            claim.insufficient_evidence = True
            claim.citation_tokens = []
            if not claim.text:
                claim.text = "证据不足"
        elif insufficient_evidence is False:
            if not claim.citation_tokens:
                raise ValueError("没有可保留的 citation token")
            claim.insufficient_evidence = False
        draft = ReviewSectionDraft.model_validate(draft.model_dump(mode="json"))
        return self._save(
            artifact,
            draft,
            expected_artifact_version=expected_artifact_version,
            created_by="user",
            model=None,
            usage={"llm_calls": 0, "operation": "claim_edit"},
            finish_reason=None,
            prompt_version="review-section-user-edit-v1",
            schema_version=REVIEW_SECTION_SCHEMA_VERSION,
        )

    def _save(
        self,
        artifact: ResearchArtifact,
        draft: ReviewSectionDraft,
        *,
        expected_artifact_version: int,
        created_by: str,
        model: Optional[str],
        usage: dict[str, Any],
        finish_reason: Optional[str],
        prompt_version: str,
        schema_version: int,
    ) -> SavedReviewSection:
        refs = []
        for claim_index, claim in enumerate(draft.claims):
            field_path = f"$.claims.{claim_index}"
            for ordinal, ref in enumerate(claim.citation_tokens):
                refs.append(
                    (
                        ref.evidence_id,
                        field_path,
                        ordinal,
                        ref.source_id,
                        ref.quote,
                    )
                )
        revision = self.repository.append_validated_review_section_revision(
            artifact.id,
            draft.model_dump(mode="json"),
            expected_artifact_version=expected_artifact_version,
            expected_project_fingerprint=self.repository.project_source_fingerprint(
                artifact.project_id
            ),
            outline_artifact_id=draft.outline_artifact_id,
            outline_revision_id=draft.outline_revision_id,
            section_key=draft.section_key,
            evidence_refs=refs,
            created_by=created_by,
            model=model,
            usage=usage,
            finish_reason=finish_reason,
            prompt_version=prompt_version,
            schema_version=schema_version,
        )
        updated = self.repository.get_artifact(artifact.id)
        if updated is None:  # pragma: no cover
            raise RuntimeError("Review section 保存后无法读取")
        return SavedReviewSection(updated, revision)


def _parse_schema(content: str, schema: type[SchemaT]) -> SchemaT:
    text = str(content).strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return schema.model_validate(json.loads(text))


def _metadata_total_tokens(metadata: dict[str, Any]) -> int:
    usage = metadata.get("usage") or {}
    if not isinstance(usage, dict):
        return 0
    total = usage.get("total_tokens")
    if isinstance(total, int) and not isinstance(total, bool):
        return max(0, total)
    return sum(
        value
        for value in (usage.get("input_tokens"), usage.get("output_tokens"))
        if isinstance(value, int) and not isinstance(value, bool) and value > 0
    )
