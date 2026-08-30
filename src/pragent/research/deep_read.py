"""有界、字段特定检索的单篇精读 map/reduce 工作流。"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, TypeVar

from pydantic import BaseModel, ValidationError

from pragent.models import Evidence
from pragent.search import search_within_paper

from .schemas import (
    DEEP_READ_FIELD_LABELS,
    DEEP_READ_FIELD_ORDER,
    DEEP_READ_SCHEMA_VERSION,
    DeepReadCard,
    DeepReadField,
)

DEEP_READ_PROMPT_VERSION = "deep-read-v2"

_FIELD_QUERIES = {
    "research_question": "research question objective problem addressed 研究问题 研究目标",
    "related_work": "related work prior studies baseline literature 相关工作 前人研究",
    "core_method": "method methodology architecture algorithm procedure 核心方法 算法",
    "contributions": "contribution novelty innovation 创新点 主要贡献",
    "datasets_and_experiments": "dataset experiment setup benchmark metrics 数据集 实验设置",
    "main_results": "results findings performance ablation 主要结果 性能 消融",
    "limitations": "limitations threats validity weakness 局限性 有效性威胁",
    "future_work": "future work further research open questions 未来工作",
    "key_evidence": "key conclusion evidence abstract conclusion 关键结论 原文证据",
}


class DeepReadError(RuntimeError):
    pass


class DeepReadBudgetExceeded(DeepReadError):
    pass


class DeepReadSchemaError(DeepReadError):
    pass


@dataclass(frozen=True)
class DeepReadBudget:
    max_retrieval_calls: int = 9
    max_llm_calls: int = 11
    max_context_chars: int = 180_000
    max_evidence: int = 45
    max_reported_tokens: int = 120_000
    hits_per_field: int = 4

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} 必须是正整数")


@dataclass
class DeepReadUsage:
    retrieval_calls: int = 0
    llm_calls: int = 0
    context_chars: int = 0
    evidence_count: int = 0
    reported_tokens: int = 0
    repair_used: bool = False


@dataclass(frozen=True)
class DeepReadFieldDraft:
    field: DeepReadField
    evidence: dict[str, Evidence]
    model: Optional[str]
    usage: dict[str, Any]
    finish_reason: Optional[str]
    prompt_version: str
    schema_version: int


@dataclass(frozen=True)
class DeepReadDraft:
    card: DeepReadCard
    evidence: dict[str, Evidence]
    model: Optional[str]
    usage: dict[str, Any]
    finish_reason: Optional[str]
    prompt_version: str
    schema_version: int


SchemaT = TypeVar("SchemaT", bound=BaseModel)


def _validation_summary(exc: Exception) -> str:
    """返回可审计但不包含模型正文/quote 的验证摘要。"""
    if isinstance(exc, ValidationError):
        items: list[str] = []
        for error in exc.errors(include_url=False, include_input=False)[:4]:
            location = ".".join(str(part) for part in error.get("loc", ())) or "$"
            error_type = str(error.get("type", "validation_error"))
            message = str(error.get("msg", "验证失败"))[:120]
            items.append(f"{location}:{error_type}:{message}")
        return "validation=" + " | ".join(items)
    if isinstance(exc, json.JSONDecodeError):
        return f"json_decode=line {exc.lineno}, column {exc.colno}: {exc.msg[:120]}"
    return f"{exc.__class__.__name__}:{str(exc)[:200]}"


class DeepReadWorkflow:
    def __init__(
        self,
        store,
        embedder,
        llm,
        *,
        budget: Optional[DeepReadBudget] = None,
        retrieval: Optional[Callable[[int, str, int], list[Any]]] = None,
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.llm = llm
        self.budget = budget or DeepReadBudget()
        self.retrieval = retrieval or (
            lambda paper_id, query, top: search_within_paper(
                store, embedder, paper_id, query, top=top
            )
        )
        self.usage = DeepReadUsage()
        self.on_progress = on_progress
        self._metadata: list[dict[str, Any]] = []

    def generate(self, paper_id: int) -> DeepReadDraft:
        paper = self.store.paper_by_id(paper_id)
        if paper is None:
            raise DeepReadError(f"索引文档不存在：{paper_id}")
        mapped: dict[str, DeepReadField] = {}
        evidence: dict[str, Evidence] = {}
        for index, field_name in enumerate(DEEP_READ_FIELD_ORDER, start=1):
            candidates = self._retrieve_field(paper_id, field_name, evidence)
            mapped[field_name] = self._map_field(field_name, candidates)
            if self.on_progress is not None:
                self.on_progress(index, len(DEEP_READ_FIELD_ORDER))
        card = self._reduce_card(mapped)
        card = self._recover_quotes(card, evidence)
        self._ensure_refs_were_retrieved(card, evidence)
        return DeepReadDraft(
            card=card,
            evidence=evidence,
            model=getattr(self.llm, "model", None),
            usage=self._aggregate_usage(),
            finish_reason=(
                self._metadata[-1].get("finish_reason") if self._metadata else None
            ),
            prompt_version=DEEP_READ_PROMPT_VERSION,
            schema_version=DEEP_READ_SCHEMA_VERSION,
        )

    def generate_field(
        self, paper_id: int, field_name: str
    ) -> DeepReadFieldDraft:
        if field_name not in DEEP_READ_FIELD_ORDER:
            raise ValueError("未知精读字段")
        evidence: dict[str, Evidence] = {}
        candidates = self._retrieve_field(paper_id, field_name, evidence)
        result = _recover_field_quotes(self._map_field(field_name, candidates), evidence)
        self._ensure_field_refs(result, evidence)
        return DeepReadFieldDraft(
            field=result,
            evidence=evidence,
            model=getattr(self.llm, "model", None),
            usage=self._aggregate_usage(),
            finish_reason=(
                self._metadata[-1].get("finish_reason") if self._metadata else None
            ),
            prompt_version=DEEP_READ_PROMPT_VERSION,
            schema_version=DEEP_READ_SCHEMA_VERSION,
        )

    def _retrieve_field(
        self,
        paper_id: int,
        field_name: str,
        all_evidence: dict[str, Evidence],
    ) -> list[Evidence]:
        self._consume("retrieval_calls", 1, self.budget.max_retrieval_calls)
        hits = self.retrieval(
            paper_id,
            _FIELD_QUERIES[field_name],
            self.budget.hits_per_field,
        )
        result: list[Evidence] = []
        for hit in hits:
            chunk_id = getattr(hit, "chunk_id", None)
            if chunk_id is None:
                continue
            pinned = self.store.pin_evidence(int(chunk_id))
            if pinned.id not in all_evidence:
                if len(all_evidence) >= self.budget.max_evidence:
                    break
                all_evidence[pinned.id] = pinned
                self.usage.evidence_count = len(all_evidence)
            if all(item.id != pinned.id for item in result):
                result.append(pinned)
        return result

    def _map_field(
        self, field_name: str, candidates: list[Evidence]
    ) -> DeepReadField:
        label = DEEP_READ_FIELD_LABELS[field_name]
        evidence_json = [
            {
                "evidence_id": item.id,
                "page": item.page,
                "text": item.text[:2500],
            }
            for item in candidates
        ]
        system = (
            "你是证据优先的论文精读助手。只使用给定证据，输出单个 JSON 对象，"
            "字段必须为 text、evidence_refs、insufficient_evidence。"
            "evidence_refs 必须是对象数组，每个元素形如 "
            '{"evidence_id": "ev_...", "quote": "逐字原文片段"}；'
            "不要输出字符串数组，也不要添加其他键。总结使用中文；"
            "quote 必须逐字复制 evidence text 中的原文，不得编造。证据不足时设置 "
            "insufficient_evidence=true、evidence_refs=[]。"
        )
        user = json.dumps(
            {"field": field_name, "label": label, "evidence": evidence_json},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        evidence_map = {item.id: item for item in candidates}

        def verify(value: DeepReadField) -> None:
            _require_quotes_locatable(value, evidence_map)

        return self._call_schema(
            system,
            user,
            DeepReadField,
            verify=verify,
            user_payload=user,
            stage=f"map:{field_name}",
        )

    def _reduce_card(self, mapped: dict[str, DeepReadField]) -> DeepReadCard:
        system = (
            "你是论文精读卡整理器。只输出符合给定九字段结构的 JSON；不得新增"
            " evidence ID、不得改写 quote。每个字段的 evidence_refs 保持输入中的"
            "对象数组形状（元素含 evidence_id 与 quote），不要改成字符串。"
            "保留证据不足标记，中文总结、英文原文不翻译。"
        )
        user = json.dumps(
            {name: value.model_dump(mode="json") for name, value in mapped.items()},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        input_quotes: dict[str, set[str]] = {}
        for value in mapped.values():
            for ref in value.evidence_refs:
                input_quotes.setdefault(ref.evidence_id, set()).add(ref.quote)

        def verify(card: DeepReadCard) -> None:
            for _name, field_value in card.ordered_fields():
                for ref in field_value.evidence_refs:
                    known = input_quotes.get(ref.evidence_id)
                    if known is not None and ref.quote not in known:
                        raise ValueError(
                            "reduce 输出改写了 quote；必须逐字保留 map 输入的原文引用"
                        )

        return self._call_schema(
            system, user, DeepReadCard, verify=verify, stage="reduce"
        )

    def _call_schema(
        self,
        system: str,
        user: str,
        schema: type[SchemaT],
        *,
        verify: Optional[Callable[[BaseModel], None]] = None,
        user_payload: Optional[str] = None,
        stage: str,
    ) -> SchemaT:
        response = self._call_llm(system, user)
        try:
            parsed = _parse_schema(response["content"], schema)
            if verify is not None:
                verify(parsed)
            return parsed
        except (ValidationError, ValueError, json.JSONDecodeError) as exc:
            if self.usage.repair_used:
                raise DeepReadSchemaError(
                    "LLM JSON schema 验证失败，repair 已用尽"
                    f"（stage={stage}; {_validation_summary(exc)}）"
                ) from exc
            self.usage.repair_used = True
            repair_system = (
                "修复下列 JSON，使其严格符合 JSON Schema。只输出 JSON，不解释；"
                "不得添加输入中不存在的 evidence ID 或 quote。"
            )
            repair_payload: dict[str, Any] = {
                "schema": schema.model_json_schema(),
                "invalid_output": str(response["content"])[:12000],
                "validation_error": str(exc)[:2000],
            }
            if user_payload is not None:
                # 带上原始证据载荷：quote 定位失败时模型必须从同一证据原文
                # 重新逐字复制，而不是凭记忆改写。
                repair_payload["original_user_payload"] = user_payload[:60_000]
            repair_user = json.dumps(
                repair_payload,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            repaired = self._call_llm(repair_system, repair_user)
            try:
                parsed = _parse_schema(repaired["content"], schema)
                if verify is not None:
                    verify(parsed)
                return parsed
            except (ValidationError, ValueError, json.JSONDecodeError) as repair_exc:
                raise DeepReadSchemaError(
                    "LLM JSON repair 后仍不符合 schema"
                    f"（stage={stage}; {_validation_summary(repair_exc)}）"
                ) from repair_exc

    def _call_llm(self, system: str, user: str) -> dict[str, Any]:
        self._consume("llm_calls", 1, self.budget.max_llm_calls)
        context_chars = len(system) + len(user)
        self._consume("context_chars", context_chars, self.budget.max_context_chars)
        if not hasattr(self.llm, "chat_with_metadata"):
            raise DeepReadError("LLM 不支持可审计 metadata 调用")
        json_call = getattr(self.llm, "chat_json_with_metadata", None)
        response = (
            json_call(system, user)
            if callable(json_call)
            else self.llm.chat_with_metadata(system, user)
        )
        if not isinstance(response, dict) or not isinstance(response.get("content"), str):
            raise DeepReadError("LLM 返回格式无效")
        metadata = response.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}
        self._metadata.append(metadata)
        reported = _metadata_total_tokens(metadata)
        self._consume(
            "reported_tokens",
            reported,
            self.budget.max_reported_tokens,
        )
        return response

    def _consume(self, name: str, amount: int, maximum: int) -> None:
        current = int(getattr(self.usage, name)) + amount
        if current > maximum:
            raise DeepReadBudgetExceeded(f"精读预算超限：{name} > {maximum}")
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
            response_id = metadata.get("response_id")
            if response_id:
                response_ids.append(str(response_id))
        return {
            **{key: value for key, value in totals.items() if value},
            "llm_calls": self.usage.llm_calls,
            "retrieval_calls": self.usage.retrieval_calls,
            "context_chars": self.usage.context_chars,
            "evidence_count": self.usage.evidence_count,
            "repair_used": self.usage.repair_used,
            "response_ids": response_ids,
        }

    @staticmethod
    def _recover_quotes(
        card: DeepReadCard, evidence: dict[str, Evidence]
    ) -> DeepReadCard:
        updates: dict[str, DeepReadField] = {}
        for name, value in card.ordered_fields():
            recovered = _recover_field_quotes(value, evidence)
            if recovered is not value:
                updates[name] = recovered
        if not updates:
            return card
        return card.model_copy(update=updates)

    @staticmethod
    def _ensure_refs_were_retrieved(
        card: DeepReadCard, evidence: dict[str, Evidence]
    ) -> None:
        for _field_name, value in card.ordered_fields():
            DeepReadWorkflow._ensure_field_refs(value, evidence)

    @staticmethod
    def _ensure_field_refs(
        value: DeepReadField, evidence: dict[str, Evidence]
    ) -> None:
        unknown = {
            item.evidence_id for item in value.evidence_refs
        } - set(evidence)
        if unknown:
            raise DeepReadSchemaError("LLM 返回了未检索到的 evidence ID")


def _parse_schema(content: str, schema: type[SchemaT]) -> SchemaT:
    text = str(content).strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return schema.model_validate(json.loads(text))


def _locate_exact_quote(text: str, quote: str) -> Optional[str]:
    """把模型转录的 quote 恢复为证据原文的精确子串；无法定位返回 None。

    模型逐字复制原文时常见三类机械漂移：空白层面（换行/多空格/全角空格）、
    PDF 断词连字符位置（原文 "perfor-\\nmance" 被转写成 "performance"）与
    兼容字形（连字/全角/数学兼容符号）。这里在忽略空白与连字符、并对两侧
    做 NFKC 归一化的意义上定位引用片段，再映射回原文字符区间，返回真正
    的原文子串；若模型改动了其他字符则定位失败，交由保存事务的精确子串
    校验 fail closed，绝不静默改写内容。
    """
    if quote in text:
        return quote
    ignorable = frozenset(" \t\r\n\f\v\u00a0\u3000-")

    def _collapsed_with_map(raw: str) -> tuple[str, list[int]]:
        chars: list[str] = []
        index_map: list[int] = []
        for i, ch in enumerate(raw):
            if ch in ignorable:
                continue
            normalized = unicodedata.normalize("NFKC", ch)
            for _ in normalized:
                index_map.append(i)
            chars.append(normalized)
        return "".join(chars), index_map

    collapsed, index_map = _collapsed_with_map(text)
    needle, _ = _collapsed_with_map(quote)
    if not needle:
        return None
    start = collapsed.find(needle)
    if start < 0:
        return None
    end = start + len(needle) - 1
    return text[index_map[start] : index_map[end] + 1]


def _recover_field_quotes(
    value: DeepReadField, evidence: dict[str, Evidence]
) -> DeepReadField:
    recovered_refs = []
    changed = False
    for ref in value.evidence_refs:
        source = evidence.get(ref.evidence_id)
        exact = (
            _locate_exact_quote(str(source.text), ref.quote)
            if source is not None
            else None
        )
        if exact is not None and exact != ref.quote:
            changed = True
            ref = ref.model_copy(update={"quote": exact})
        recovered_refs.append(ref)
    if not changed:
        return value
    return value.model_copy(update={"evidence_refs": recovered_refs})


def _require_quotes_locatable(
    value: DeepReadField, evidence: Mapping[str, Evidence]
) -> None:
    """校验字段引用的 quote 可在证据原文中定位（恢复后）。

    无法定位说明模型改写了原文内容；作为 schema 层错误抛出以触发一次
    有界 repair（repair 载荷携带原始证据文本，模型可重新逐字复制）。
    """
    for ref in value.evidence_refs:
        source = evidence.get(ref.evidence_id)
        if source is None:
            raise ValueError(f"quote 引用了未检索到的证据：{ref.evidence_id}")
        if _locate_exact_quote(str(source.text), ref.quote) is None:
            raise ValueError(
                "evidence_refs 中的 quote 无法在证据原文中定位；"
                "必须从 original_user_payload 的 evidence text 逐字复制，"
                "或在证据不足时设置 insufficient_evidence=true 且清空 refs"
            )


def _metadata_total_tokens(metadata: dict[str, Any]) -> int:
    usage = metadata.get("usage") or {}
    if not isinstance(usage, dict):
        return 0
    total = usage.get("total_tokens")
    if isinstance(total, int) and not isinstance(total, bool):
        return max(0, total)
    values = [usage.get("input_tokens"), usage.get("output_tokens")]
    return sum(
        value
        for value in values
        if isinstance(value, int) and not isinstance(value, bool) and value > 0
    )
