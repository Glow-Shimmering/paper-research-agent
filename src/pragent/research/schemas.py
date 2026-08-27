"""单篇精读的稳定 Pydantic 合同。"""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

DEEP_READ_SCHEMA_VERSION = 1
COMPARISON_SCHEMA_VERSION = 1
REVIEW_OUTLINE_SCHEMA_VERSION = 1
DEEP_READ_FIELD_ORDER = (
    "research_question",
    "related_work",
    "core_method",
    "contributions",
    "datasets_and_experiments",
    "main_results",
    "limitations",
    "future_work",
    "key_evidence",
)
DEEP_READ_FIELD_LABELS = {
    "research_question": "研究问题",
    "related_work": "相关工作",
    "core_method": "核心方法",
    "contributions": "创新点",
    "datasets_and_experiments": "数据集与实验",
    "main_results": "主要结果",
    "limitations": "局限性",
    "future_work": "未来工作",
    "key_evidence": "关键原文证据",
}


class EvidenceRef(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    evidence_id: str = Field(min_length=4, max_length=96, pattern=r"^ev_[0-9a-f]+$")
    quote: str = Field(min_length=1, max_length=4000)


class DeepReadField(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    text: str = Field(default="", max_length=12000)
    evidence_refs: list[EvidenceRef] = Field(default_factory=list, max_length=20)
    insufficient_evidence: bool = False

    @model_validator(mode="after")
    def validate_support_state(self) -> "DeepReadField":
        if self.insufficient_evidence:
            if self.evidence_refs:
                raise ValueError("证据不足字段不能同时声明 evidence_refs")
            return self
        if self.text and not self.evidence_refs:
            raise ValueError("非空精读字段必须包含 evidence_refs")
        if not self.text:
            raise ValueError("空字段必须明确 insufficient_evidence=true")
        identities = [item.evidence_id for item in self.evidence_refs]
        if len(identities) != len(set(identities)):
            raise ValueError("同一字段不能重复引用 evidence")
        return self


class DeepReadCard(BaseModel):
    model_config = ConfigDict(extra="forbid")
    field_order: ClassVar[tuple[str, ...]] = DEEP_READ_FIELD_ORDER

    research_question: DeepReadField
    related_work: DeepReadField
    core_method: DeepReadField
    contributions: DeepReadField
    datasets_and_experiments: DeepReadField
    main_results: DeepReadField
    limitations: DeepReadField
    future_work: DeepReadField
    key_evidence: DeepReadField

    def ordered_fields(self) -> tuple[tuple[str, DeepReadField], ...]:
        return tuple((name, getattr(self, name)) for name in self.field_order)


class ComparisonDimension(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    key: str = Field(min_length=2, max_length=64, pattern=r"^[a-z][a-z0-9_]*$")
    label: str = Field(min_length=1, max_length=80)
    description: str = Field(default="", max_length=500)
    source_field: str | None = None

    @model_validator(mode="after")
    def validate_source_field(self) -> "ComparisonDimension":
        if self.source_field is not None and self.source_field not in DEEP_READ_FIELD_ORDER:
            raise ValueError("source_field 必须是精读卡固定字段")
        return self


class ComparisonCell(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    source_id: str = Field(min_length=1, max_length=96)
    dimension_key: str = Field(
        min_length=2, max_length=64, pattern=r"^[a-z][a-z0-9_]*$"
    )
    summary: str = Field(default="", max_length=12000)
    evidence_refs: list[EvidenceRef] = Field(default_factory=list, max_length=20)
    insufficient_evidence: bool = False

    @model_validator(mode="after")
    def validate_support_state(self) -> "ComparisonCell":
        if self.insufficient_evidence:
            if self.evidence_refs:
                raise ValueError("证据不足 cell 不能同时声明 evidence_refs")
            return self
        if not self.summary:
            raise ValueError("有证据的 comparison cell 必须包含 summary")
        if not self.evidence_refs:
            raise ValueError("有证据的 comparison cell 必须包含 evidence_refs")
        identities = [item.evidence_id for item in self.evidence_refs]
        if len(identities) != len(set(identities)):
            raise ValueError("同一 comparison cell 不能重复引用 evidence")
        return self


class ComparisonDimensionCells(BaseModel):
    """自定义维度的单次 LLM 输出合同。"""

    model_config = ConfigDict(extra="forbid")

    dimension_key: str = Field(
        min_length=2, max_length=64, pattern=r"^[a-z][a-z0-9_]*$"
    )
    cells: list[ComparisonCell] = Field(min_length=2, max_length=20)


class ComparisonMatrix(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=200)
    source_ids: list[str] = Field(min_length=2, max_length=20)
    dimensions: list[ComparisonDimension] = Field(min_length=1, max_length=40)
    cells: list[ComparisonCell] = Field(min_length=2, max_length=800)

    @model_validator(mode="after")
    def validate_matrix_shape(self) -> "ComparisonMatrix":
        if len(self.source_ids) != len(set(self.source_ids)):
            raise ValueError("comparison source_ids 不能重复")
        dimension_keys = [item.key for item in self.dimensions]
        if len(dimension_keys) != len(set(dimension_keys)):
            raise ValueError("comparison dimension key 不能重复")
        pairs = [(cell.source_id, cell.dimension_key) for cell in self.cells]
        if len(pairs) != len(set(pairs)):
            raise ValueError("comparison cell 不能重复")
        expected = {
            (source_id, dimension_key)
            for source_id in self.source_ids
            for dimension_key in dimension_keys
        }
        if set(pairs) != expected:
            raise ValueError("comparison cells 必须完整覆盖 source × dimension")
        return self


class ReviewSourceEvidenceRef(EvidenceRef):
    source_id: str = Field(min_length=1, max_length=96)


class ReviewOutlineClaim(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    text: str = Field(min_length=1, max_length=2000)
    source_ids: list[str] = Field(min_length=1, max_length=20)
    evidence_refs: list[ReviewSourceEvidenceRef] = Field(
        default_factory=list, max_length=40
    )
    insufficient_evidence: bool = False

    @model_validator(mode="after")
    def validate_support_state(self) -> "ReviewOutlineClaim":
        if len(self.source_ids) != len(set(self.source_ids)):
            raise ValueError("综述计划 claim 的 source_ids 不能重复")
        if self.insufficient_evidence:
            if self.evidence_refs:
                raise ValueError("证据不足 claim 不能同时声明 evidence_refs")
            return self
        if not self.evidence_refs:
            raise ValueError("有证据的综述计划 claim 必须包含 evidence_refs")
        evidence_sources = [item.source_id for item in self.evidence_refs]
        if set(evidence_sources) != set(self.source_ids):
            raise ValueError("claim 的每个来源都必须有 evidence 支持")
        identities = [
            (item.source_id, item.evidence_id) for item in self.evidence_refs
        ]
        if len(identities) != len(set(identities)):
            raise ValueError("同一 claim 不能重复引用来源 evidence")
        return self


class ReviewOutlineSection(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    key: str = Field(min_length=2, max_length=64, pattern=r"^[a-z][a-z0-9_]*$")
    title: str = Field(min_length=1, max_length=200)
    objective: str = Field(min_length=1, max_length=2000)
    source_ids: list[str] = Field(min_length=1, max_length=20)
    planned_claims: list[ReviewOutlineClaim] = Field(min_length=1, max_length=30)

    @model_validator(mode="after")
    def validate_source_scope(self) -> "ReviewOutlineSection":
        if len(self.source_ids) != len(set(self.source_ids)):
            raise ValueError("综述 section 的 source_ids 不能重复")
        claim_sources = {
            source_id
            for claim in self.planned_claims
            for source_id in claim.source_ids
        }
        if claim_sources != set(self.source_ids):
            raise ValueError("section source_ids 必须等于 planned claims 来源并集")
        return self


class ReviewOutlinePayload(BaseModel):
    """LLM 仅负责生成标题与章节，不允许生成输入 provenance。"""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=200)
    sections: list[ReviewOutlineSection] = Field(min_length=2, max_length=20)

    @model_validator(mode="after")
    def validate_unique_sections(self) -> "ReviewOutlinePayload":
        keys = [item.key for item in self.sections]
        if len(keys) != len(set(keys)):
            raise ValueError("综述 section key 不能重复")
        return self


class ReviewQuestionSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    id: str = Field(min_length=1, max_length=96)
    question: str = Field(min_length=1, max_length=1000)
    version: int = Field(ge=1)


class ReviewOutline(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=200)
    research_questions: list[ReviewQuestionSnapshot] = Field(
        min_length=1, max_length=20
    )
    source_ids: list[str] = Field(min_length=2, max_length=20)
    comparison_artifact_id: str = Field(min_length=1, max_length=96)
    comparison_revision_id: str = Field(min_length=1, max_length=96)
    sections: list[ReviewOutlineSection] = Field(min_length=2, max_length=20)

    @model_validator(mode="after")
    def validate_scope(self) -> "ReviewOutline":
        question_ids = [item.id for item in self.research_questions]
        if len(question_ids) != len(set(question_ids)):
            raise ValueError("综述 research question 不能重复")
        if len(self.source_ids) != len(set(self.source_ids)):
            raise ValueError("综述 source_ids 不能重复")
        selected = set(self.source_ids)
        if any(not set(section.source_ids) <= selected for section in self.sections):
            raise ValueError("综述 section 引用了未选择来源")
        return self
