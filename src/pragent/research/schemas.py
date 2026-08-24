"""单篇精读的稳定 Pydantic 合同。"""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

DEEP_READ_SCHEMA_VERSION = 1
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
