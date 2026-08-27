"""Project-scoped comparison workflow built only from current Deep Read cards."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional, TypeVar

from pydantic import BaseModel, ValidationError

from pragent.models import ArtifactRevision, ResearchArtifact
from pragent.storage import ResearchRepository

from .schemas import (
    COMPARISON_SCHEMA_VERSION,
    DEEP_READ_FIELD_LABELS,
    DEEP_READ_FIELD_ORDER,
    ComparisonCell,
    ComparisonDimension,
    ComparisonDimensionCells,
    ComparisonMatrix,
    DeepReadCard,
)

COMPARISON_PROMPT_VERSION = "comparison-v1"


class ComparisonError(RuntimeError):
    pass


class ComparisonBudgetExceeded(ComparisonError):
    pass


class ComparisonSchemaError(ComparisonError):
    pass


class ComparisonPrerequisiteError(ComparisonError):
    def __init__(
        self,
        *,
        missing_source_ids: Iterable[str] = (),
        stale_source_ids: Iterable[str] = (),
    ) -> None:
        self.missing_source_ids = tuple(missing_source_ids)
        self.stale_source_ids = tuple(stale_source_ids)
        parts = []
        if self.missing_source_ids:
            parts.append("缺少精读卡：" + ", ".join(self.missing_source_ids))
        if self.stale_source_ids:
            parts.append("精读卡已过期：" + ", ".join(self.stale_source_ids))
        super().__init__("；".join(parts) or "比较工作流前置条件未满足")


@dataclass(frozen=True)
class ComparisonBudget:
    max_sources: int = 20
    max_custom_dimensions: int = 20
    max_llm_calls: int = 21
    max_context_chars: int = 500_000
    max_reported_tokens: int = 200_000

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} 必须是正整数")
        if self.max_sources > 20:
            raise ValueError("max_sources 不能超过产品上限 20")


@dataclass
class ComparisonUsage:
    llm_calls: int = 0
    context_chars: int = 0
    reported_tokens: int = 0
    repair_used: bool = False


@dataclass(frozen=True)
class ComparisonDraft:
    matrix: ComparisonMatrix
    model: Optional[str]
    usage: dict[str, Any]
    finish_reason: Optional[str]
    prompt_version: str
    schema_version: int


@dataclass(frozen=True)
class SavedComparison:
    artifact: ResearchArtifact
    revision: ArtifactRevision


SchemaT = TypeVar("SchemaT", bound=BaseModel)


def default_comparison_dimensions() -> tuple[ComparisonDimension, ...]:
    return tuple(
        ComparisonDimension(
            key=field_name,
            label=DEEP_READ_FIELD_LABELS[field_name],
            source_field=field_name,
        )
        for field_name in DEEP_READ_FIELD_ORDER
    )


class ComparisonWorkflow:
    def __init__(
        self,
        repository: ResearchRepository,
        llm=None,
        *,
        budget: Optional[ComparisonBudget] = None,
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        self.repository = repository
        self.llm = llm
        self.budget = budget or ComparisonBudget()
        self.on_progress = on_progress
        self.usage = ComparisonUsage()
        self._metadata: list[dict[str, Any]] = []

    def generate(
        self,
        project_id: str,
        source_ids: Iterable[str],
        *,
        title: str = "跨论文比较矩阵",
        custom_dimensions: Iterable[ComparisonDimension] = (),
    ) -> ComparisonDraft:
        selected = tuple(str(item).strip() for item in source_ids)
        self._validate_sources(project_id, selected)
        custom = self._validate_custom_dimensions(tuple(custom_dimensions))
        cards, source_titles = self._load_current_cards(project_id, selected)
        dimensions = default_comparison_dimensions() + custom
        cells = self._default_cells(selected, cards)
        total = len(custom)
        for index, dimension in enumerate(custom, start=1):
            cells.extend(
                self._generate_custom_dimension(
                    dimension, selected, cards, source_titles
                )
            )
            if self.on_progress is not None:
                self.on_progress(index, total)
        matrix = ComparisonMatrix(
            title=str(title).strip() or "跨论文比较矩阵",
            source_ids=list(selected),
            dimensions=list(dimensions),
            cells=cells,
        )
        return ComparisonDraft(
            matrix=matrix,
            model=(getattr(self.llm, "model", None) if custom else None),
            usage=self._aggregate_usage(),
            finish_reason=(
                self._metadata[-1].get("finish_reason") if self._metadata else None
            ),
            prompt_version=COMPARISON_PROMPT_VERSION,
            schema_version=COMPARISON_SCHEMA_VERSION,
        )

    def _validate_sources(self, project_id: str, source_ids: tuple[str, ...]) -> None:
        if self.repository.get_project(project_id) is None:
            raise KeyError(f"研究项目不存在：{project_id}")
        if not 2 <= len(source_ids) <= self.budget.max_sources:
            raise ValueError("比较工作流必须选择 2–20 个来源")
        if any(not item for item in source_ids) or len(source_ids) != len(set(source_ids)):
            raise ValueError("比较来源不能为空或重复")
        memberships = {
            item.source.id
            for item in self.repository.list_project_sources(
                project_id, limit=200
            ).items
        }
        outside = set(source_ids) - memberships
        if outside:
            raise ValueError("比较来源必须全部属于当前项目")

    def _validate_custom_dimensions(
        self, dimensions: tuple[ComparisonDimension, ...]
    ) -> tuple[ComparisonDimension, ...]:
        if len(dimensions) > self.budget.max_custom_dimensions:
            raise ValueError("自定义比较维度超过预算")
        default_keys = set(DEEP_READ_FIELD_ORDER)
        keys = [item.key for item in dimensions]
        if len(keys) != len(set(keys)) or default_keys & set(keys):
            raise ValueError("自定义维度 key 不能重复或覆盖默认维度")
        if any(item.source_field is not None for item in dimensions):
            raise ValueError("自定义维度不能伪装为精读卡固定字段")
        if dimensions and (
            self.llm is None or not hasattr(self.llm, "chat_with_metadata")
        ):
            raise ComparisonError("自定义比较维度需要可审计 metadata 的 LLM")
        return dimensions

    def _load_current_cards(
        self, project_id: str, source_ids: tuple[str, ...]
    ) -> tuple[dict[str, DeepReadCard], dict[str, str]]:
        cards: dict[str, DeepReadCard] = {}
        titles: dict[str, str] = {}
        missing: list[str] = []
        stale: list[str] = []
        memberships = {
            item.source.id: item.source
            for item in self.repository.list_project_sources(
                project_id, limit=200
            ).items
        }
        for source_id in source_ids:
            source = memberships[source_id]
            titles[source_id] = source.title or source_id
            artifact = self.repository.get_source_artifact(
                project_id, source_id, "deep_read"
            )
            if artifact is None or artifact.current_revision_number < 1:
                missing.append(source_id)
                continue
            freshness = self.repository.artifact_freshness(artifact.id)
            if artifact.status != "ready" or freshness.stale:
                stale.append(source_id)
                continue
            revision = self.repository.get_current_artifact_revision(artifact.id)
            if revision is None:
                missing.append(source_id)
                continue
            try:
                cards[source_id] = DeepReadCard.model_validate(revision.content)
            except ValidationError as exc:
                raise ComparisonPrerequisiteError(
                    stale_source_ids=(source_id,)
                ) from exc
        if missing or stale:
            raise ComparisonPrerequisiteError(
                missing_source_ids=missing,
                stale_source_ids=stale,
            )
        return cards, titles

    @staticmethod
    def _default_cells(
        source_ids: tuple[str, ...], cards: dict[str, DeepReadCard]
    ) -> list[ComparisonCell]:
        cells: list[ComparisonCell] = []
        for source_id in source_ids:
            card = cards[source_id]
            for field_name, field in card.ordered_fields():
                cells.append(
                    ComparisonCell(
                        source_id=source_id,
                        dimension_key=field_name,
                        summary=field.text,
                        evidence_refs=field.evidence_refs,
                        insufficient_evidence=field.insufficient_evidence,
                    )
                )
        return cells

    def _generate_custom_dimension(
        self,
        dimension: ComparisonDimension,
        source_ids: tuple[str, ...],
        cards: dict[str, DeepReadCard],
        source_titles: dict[str, str],
    ) -> list[ComparisonCell]:
        source_payload = []
        allowed_quotes: dict[str, dict[str, set[str]]] = {}
        for source_id in source_ids:
            card = cards[source_id]
            quote_map: dict[str, set[str]] = {}
            for _name, field in card.ordered_fields():
                for ref in field.evidence_refs:
                    quote_map.setdefault(ref.evidence_id, set()).add(ref.quote)
            allowed_quotes[source_id] = quote_map
            source_payload.append(
                {
                    "source_id": source_id,
                    "title": source_titles[source_id],
                    "deep_read": card.model_dump(mode="json"),
                }
            )
        system = (
            "你是证据优先的跨论文比较助手。只为给定自定义维度输出 JSON；"
            "每个 source 必须恰好一个 cell。只能复用该 source 精读卡中已有的 "
            "evidence_id 与逐字 quote，不得跨来源引用或创造证据。证据不足时设置 "
            "insufficient_evidence=true、evidence_refs=[]。"
        )
        user = json.dumps(
            {
                "dimension": dimension.model_dump(mode="json"),
                "sources": source_payload,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        result = self._call_schema(system, user, ComparisonDimensionCells)
        if result.dimension_key != dimension.key:
            raise ComparisonSchemaError("LLM 返回的自定义维度 key 不匹配")
        returned = [cell.source_id for cell in result.cells]
        if len(returned) != len(set(returned)) or set(returned) != set(source_ids):
            raise ComparisonSchemaError("自定义维度必须恰好覆盖所选来源")
        for cell in result.cells:
            if cell.dimension_key != dimension.key:
                raise ComparisonSchemaError("自定义 cell dimension_key 不匹配")
            for ref in cell.evidence_refs:
                quotes = allowed_quotes[cell.source_id].get(ref.evidence_id)
                if quotes is None or ref.quote not in quotes:
                    raise ComparisonSchemaError("自定义维度引用了越界 evidence 或改写 quote")
        by_source = {cell.source_id: cell for cell in result.cells}
        return [by_source[source_id] for source_id in source_ids]

    def _call_schema(self, system: str, user: str, schema: type[SchemaT]) -> SchemaT:
        response = self._call_llm(system, user)
        try:
            return _parse_schema(response["content"], schema)
        except (ValidationError, ValueError, json.JSONDecodeError) as exc:
            if self.usage.repair_used:
                raise ComparisonSchemaError("比较 JSON schema 验证失败，repair 已用尽") from exc
            self.usage.repair_used = True
            repaired = self._call_llm(
                "修复下列 JSON，使其严格符合 JSON Schema。只输出 JSON；不得添加输入中不存在的 evidence ID 或 quote。",
                json.dumps(
                    {
                        "schema": schema.model_json_schema(),
                        "invalid_output": str(response["content"])[:20000],
                        "validation_error": str(exc)[:3000],
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            )
            try:
                return _parse_schema(repaired["content"], schema)
            except (ValidationError, ValueError, json.JSONDecodeError) as repair_exc:
                raise ComparisonSchemaError("比较 JSON repair 后仍不符合 schema") from repair_exc

    def _call_llm(self, system: str, user: str) -> dict[str, Any]:
        self._consume("llm_calls", 1, self.budget.max_llm_calls)
        self._consume(
            "context_chars",
            len(system) + len(user),
            self.budget.max_context_chars,
        )
        response = self.llm.chat_with_metadata(system, user)
        if not isinstance(response, dict) or not isinstance(response.get("content"), str):
            raise ComparisonError("LLM 返回格式无效")
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
            raise ComparisonBudgetExceeded(f"比较预算超限：{name} > {maximum}")
        setattr(self.usage, name, current)

    def _aggregate_usage(self) -> dict[str, Any]:
        token_keys = (
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "input_tokens",
            "output_tokens",
        )
        totals = {key: 0 for key in token_keys}
        response_ids: list[str] = []
        for metadata in self._metadata:
            usage = metadata.get("usage") or {}
            if isinstance(usage, dict):
                for key in token_keys:
                    value = usage.get(key)
                    if isinstance(value, int) and not isinstance(value, bool):
                        totals[key] += value
            if metadata.get("response_id"):
                response_ids.append(str(metadata["response_id"]))
        return {
            **{key: value for key, value in totals.items() if value},
            "llm_calls": self.usage.llm_calls,
            "context_chars": self.usage.context_chars,
            "repair_used": self.usage.repair_used,
            "response_ids": response_ids,
        }


class ComparisonArtifactService:
    def __init__(self, repository: ResearchRepository) -> None:
        self.repository = repository

    def generate_and_save(
        self,
        project_id: str,
        source_ids: Iterable[str],
        workflow: ComparisonWorkflow,
        *,
        title: str = "跨论文比较矩阵",
        custom_dimensions: Iterable[ComparisonDimension] = (),
    ) -> SavedComparison:
        artifact = self.repository.create_artifact(
            project_id,
            "comparison",
            title=str(title).strip() or "跨论文比较矩阵",
            status="generating",
        )
        freshness = self.repository.artifact_freshness(artifact.id)
        if not freshness.current_fingerprint:
            raise ComparisonError("无法计算项目来源 fingerprint")
        draft = workflow.generate(
            project_id,
            source_ids,
            title=title,
            custom_dimensions=custom_dimensions,
        )
        refs = []
        for cell in draft.matrix.cells:
            field_path = f"$.cells.{cell.source_id}.{cell.dimension_key}"
            for ordinal, ref in enumerate(cell.evidence_refs):
                refs.append(
                    (
                        ref.evidence_id,
                        field_path,
                        ordinal,
                        cell.source_id,
                        ref.quote,
                    )
                )
        revision = self.repository.append_validated_comparison_revision(
            artifact.id,
            draft.matrix.model_dump(mode="json"),
            expected_artifact_version=artifact.version,
            expected_project_fingerprint=freshness.current_fingerprint,
            selected_source_ids=draft.matrix.source_ids,
            evidence_refs=refs,
            created_by="model" if draft.usage["llm_calls"] else "system",
            model=draft.model,
            usage=draft.usage,
            finish_reason=draft.finish_reason,
            prompt_version=draft.prompt_version,
            schema_version=draft.schema_version,
        )
        updated = self.repository.get_artifact(artifact.id)
        if updated is None:  # pragma: no cover
            raise RuntimeError("Comparison artifact 保存后无法读取")
        return SavedComparison(updated, revision)


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
