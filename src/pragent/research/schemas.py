"""单篇精读的稳定 Pydantic 合同。"""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

DEEP_READ_SCHEMA_VERSION = 1
COMPARISON_SCHEMA_VERSION = 1
REVIEW_OUTLINE_SCHEMA_VERSION = 1
REVIEW_SECTION_SCHEMA_VERSION = 1
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


def _dedupe_evidence_ref_values(values: object) -> object:
    """按引用身份确定性去重，保留首次出现。

    真实模型常在同一字段/cell/claim 内对同一证据多次引用（对不同论断
    复用同一支持证据）。引用身份相同即同一支持证据，保留第一次即可，
    不应因此消耗宝贵的 repair 预算；去重不新增、不改写任何内容。
    """
    if not isinstance(values, list):
        return values
    seen: set[tuple] = set()
    deduped: list = []
    for item in values:
        if isinstance(item, dict):
            key = (item.get("source_id"), item.get("evidence_id"))
        else:
            key = (
                getattr(item, "source_id", None),
                getattr(item, "evidence_id", None),
            )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _prefer_explicit_support(values: object, refs_key: str) -> object:
    """有引用时确定性消解模型输出的证据状态矛盾。

    仅把 ``insufficient_evidence`` 从 true 纠正为 false；引用随后仍需通过
    schema、来源范围和逐字 quote 校验，因此不会把无效证据变成有效证据。
    """
    if not isinstance(values, dict):
        return values
    if values.get("insufficient_evidence") is True and values.get(refs_key):
        normalized = dict(values)
        normalized["insufficient_evidence"] = False
        return normalized
    return values


class DeepReadField(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    text: str = Field(default="", max_length=12000)
    evidence_refs: list[EvidenceRef] = Field(default_factory=list, max_length=20)
    insufficient_evidence: bool = False

    @model_validator(mode="before")
    @classmethod
    def _normalize_support_state(cls, values: object) -> object:
        return _prefer_explicit_support(values, "evidence_refs")

    @field_validator("evidence_refs", mode="before")
    @classmethod
    def _dedupe_refs(cls, values: object) -> object:
        return _dedupe_evidence_ref_values(values)

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

    @model_validator(mode="before")
    @classmethod
    def _normalize_support_state(cls, values: object) -> object:
        return _prefer_explicit_support(values, "evidence_refs")

    @field_validator("evidence_refs", mode="before")
    @classmethod
    def _dedupe_refs(cls, values: object) -> object:
        return _dedupe_evidence_ref_values(values)

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

    @model_validator(mode="before")
    @classmethod
    def _normalize_support_state(cls, values: object) -> object:
        return _prefer_explicit_support(values, "evidence_refs")

    @field_validator("evidence_refs", mode="before")
    @classmethod
    def _dedupe_refs(cls, values: object) -> object:
        return _dedupe_evidence_ref_values(values)

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


class ReviewDraftClaim(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    key: str = Field(min_length=2, max_length=64, pattern=r"^[a-z][a-z0-9_]*$")
    text: str = Field(min_length=1, max_length=12000)
    citation_tokens: list[ReviewSourceEvidenceRef] = Field(
        default_factory=list, max_length=40
    )
    insufficient_evidence: bool = False

    @model_validator(mode="before")
    @classmethod
    def _normalize_support_state(cls, values: object) -> object:
        return _prefer_explicit_support(values, "citation_tokens")

    @field_validator("citation_tokens", mode="before")
    @classmethod
    def _dedupe_citation_tokens(cls, values: object) -> object:
        return _dedupe_evidence_ref_values(values)

    @model_validator(mode="after")
    def validate_citations(self) -> "ReviewDraftClaim":
        if self.insufficient_evidence:
            if self.citation_tokens:
                raise ValueError("证据不足的 section claim 不能包含 citation tokens")
            return self
        if not self.citation_tokens:
            raise ValueError("section claim 必须包含 citation tokens 或明确证据不足")
        return self


class ReviewSectionPayload(BaseModel):
    """LLM 输出合同；artifact/outline provenance 由系统包装。"""

    model_config = ConfigDict(extra="forbid")

    claims: list[ReviewDraftClaim] = Field(min_length=1, max_length=50)

    @model_validator(mode="after")
    def validate_unique_claims(self) -> "ReviewSectionPayload":
        keys = [item.key for item in self.claims]
        if len(keys) != len(set(keys)):
            raise ValueError("section draft claim key 不能重复")
        return self


class ReviewSectionDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    outline_artifact_id: str = Field(min_length=1, max_length=96)
    outline_revision_id: str = Field(min_length=1, max_length=96)
    section_key: str = Field(
        min_length=2, max_length=64, pattern=r"^[a-z][a-z0-9_]*$"
    )
    section_title: str = Field(min_length=1, max_length=200)
    claims: list[ReviewDraftClaim] = Field(min_length=1, max_length=50)

    @model_validator(mode="after")
    def validate_unique_claims(self) -> "ReviewSectionDraft":
        keys = [item.key for item in self.claims]
        if len(keys) != len(set(keys)):
            raise ValueError("section draft claim key 不能重复")
        return self
