"""受控 Agent 运行时与统一答案验证。

本模块刻意不依赖具体 LLM、工具或存储实现。它负责三件事：

* 明确、可审计的运行状态机；
* planner / executor / verifier 的最小预算边界；
* ``ask`` 与 ``chat`` 共用的引用验证规则。
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Iterable, Mapping, Optional, Sequence


class RunStatus(str, Enum):
    PROPOSED = "proposed"
    RUNNING = "running"
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"


TERMINAL_STATUSES = frozenset(
    {RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.CANCELLED, RunStatus.BLOCKED}
)

_ALLOWED_TRANSITIONS: Mapping[RunStatus, frozenset[RunStatus]] = {
    RunStatus.PROPOSED: frozenset(
        {RunStatus.RUNNING, RunStatus.CANCELLED, RunStatus.BLOCKED}
    ),
    RunStatus.RUNNING: frozenset(
        {
            RunStatus.AWAITING_CONFIRMATION,
            RunStatus.SUCCEEDED,
            RunStatus.FAILED,
            RunStatus.CANCELLED,
            RunStatus.BLOCKED,
        }
    ),
    RunStatus.AWAITING_CONFIRMATION: frozenset(
        {RunStatus.RUNNING, RunStatus.FAILED, RunStatus.CANCELLED, RunStatus.BLOCKED}
    ),
    RunStatus.SUCCEEDED: frozenset(),
    RunStatus.FAILED: frozenset(),
    RunStatus.CANCELLED: frozenset(),
    RunStatus.BLOCKED: frozenset(),
}


class InvalidRunTransition(ValueError):
    """运行状态转换不在显式状态图中。"""


class AgentBudgetExceeded(RuntimeError):
    """Agent 已到达一个硬预算边界。"""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def coerce_run_status(value: RunStatus | str) -> RunStatus:
    try:
        return value if isinstance(value, RunStatus) else RunStatus(str(value))
    except ValueError as exc:
        raise InvalidRunTransition(f"未知运行状态：{value!r}") from exc


def validate_run_transition(
    current: RunStatus | str, target: RunStatus | str
) -> tuple[RunStatus, RunStatus]:
    """校验并规范化一次状态转换；同状态写入也视为非法。"""
    source = coerce_run_status(current)
    destination = coerce_run_status(target)
    if destination not in _ALLOWED_TRANSITIONS[source]:
        raise InvalidRunTransition(
            f"非法 Agent 状态转换：{source.value} -> {destination.value}"
        )
    return source, destination


@dataclass(frozen=True)
class AgentBudget:
    """单次运行的硬上限；所有值都会持久化到 run 记录。"""

    max_rounds: int = 10
    max_tool_calls: int = 20
    max_external_calls: int = 3
    max_replans: int = 1
    max_verification_repairs: int = 1

    def __post_init__(self) -> None:
        positive = {
            "max_rounds": self.max_rounds,
            "max_tool_calls": self.max_tool_calls,
        }
        non_negative = {
            "max_external_calls": self.max_external_calls,
            "max_replans": self.max_replans,
            "max_verification_repairs": self.max_verification_repairs,
        }
        for name, value in positive.items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} 必须是正整数")
        for name, value in non_negative.items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} 必须是非负整数")

    @classmethod
    def from_value(cls, value: Optional["AgentBudget | Mapping[str, object]"]) -> "AgentBudget":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("budget 必须是 AgentBudget 或映射")
        allowed = {field.name for field in cls.__dataclass_fields__.values()}
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(f"未知预算字段：{', '.join(unknown)}")
        return cls(**{key: int(raw) for key, raw in value.items()})

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class AgentPlan:
    version: int
    objective: str
    steps: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {"version": self.version, "objective": self.objective, "steps": list(self.steps)}


class Planner:
    """确定性的最小 planner；不让模型生成无界计划。"""

    DEFAULT_STEPS = ("理解目标", "执行必要且获授权的工具", "验证并交付答案")

    def __init__(self, *, max_steps: int = 8):
        if max_steps < 1:
            raise ValueError("max_steps 必须至少为 1")
        self.max_steps = max_steps

    def create_plan(
        self, objective: str, steps: Optional[Sequence[str]] = None
    ) -> AgentPlan:
        normalized_objective = str(objective or "").strip()
        if not normalized_objective:
            raise ValueError("Agent objective 不能为空")
        source = self.DEFAULT_STEPS if steps is None else steps
        normalized_steps = tuple(str(step).strip() for step in source if str(step).strip())
        if not normalized_steps:
            raise ValueError("Agent plan 至少需要一个步骤")
        if len(normalized_steps) > self.max_steps:
            raise ValueError(f"Agent plan 最多允许 {self.max_steps} 个步骤")
        return AgentPlan(version=1, objective=normalized_objective, steps=normalized_steps)


@dataclass
class AgentRuntime:
    """一次运行的受控、纯内存状态；Store 由调用方镜像持久化。"""

    plan: AgentPlan
    budget: AgentBudget = field(default_factory=AgentBudget)
    status: RunStatus = RunStatus.PROPOSED
    round_count: int = 0
    tool_call_count: int = 0
    external_call_count: int = 0
    replan_count: int = 0
    verification_repair_count: int = 0
    step_index: int = 0
    evidence_ids: list[str] = field(default_factory=list)
    last_error: Optional[str] = None
    fatal_tool_failure: bool = False

    @property
    def objective(self) -> str:
        return self.plan.objective

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES

    @property
    def current_step(self) -> Optional[str]:
        if self.step_index >= len(self.plan.steps):
            return None
        return self.plan.steps[self.step_index]

    def transition(self, target: RunStatus | str, *, error: Optional[str] = None) -> RunStatus:
        _, destination = validate_run_transition(self.status, target)
        self.status = destination
        if error:
            self.last_error = str(error)
        return destination

    def start(self) -> None:
        self.transition(RunStatus.RUNNING)

    def begin_round(self) -> int:
        self._require_running()
        if self.round_count >= self.budget.max_rounds:
            self.transition(
                RunStatus.FAILED,
                error=f"LLM 轮次超过上限 {self.budget.max_rounds}",
            )
            raise AgentBudgetExceeded("round_budget_exceeded", self.last_error or "轮次超限")
        self.round_count += 1
        return self.round_count

    def authorize_tool(self, name: str, *, external: bool = False) -> None:
        self._require_running()
        if self.fatal_tool_failure:
            self.transition(RunStatus.FAILED, error=self.last_error or "工具失败后禁止继续执行")
            raise AgentBudgetExceeded(
                "tool_after_terminal_failure", self.last_error or "工具失败后禁止继续执行"
            )
        if self.tool_call_count >= self.budget.max_tool_calls:
            self.transition(
                RunStatus.BLOCKED,
                error=f"工具调用超过上限 {self.budget.max_tool_calls}",
            )
            raise AgentBudgetExceeded("tool_budget_exceeded", self.last_error or "工具调用超限")
        if external and self.external_call_count >= self.budget.max_external_calls:
            self.transition(
                RunStatus.BLOCKED,
                error=f"外部调用超过上限 {self.budget.max_external_calls}",
            )
            raise AgentBudgetExceeded(
                "external_call_budget_exceeded", self.last_error or "外部调用超限"
            )
        self.tool_call_count += 1
        if external:
            self.external_call_count += 1

    def observe_tool_result(
        self,
        *,
        ok: bool,
        retryable: bool = False,
        evidence_ids: Iterable[object] = (),
        message: Optional[str] = None,
    ) -> str:
        """记录执行结果，返回 ``continue`` / ``replan`` / ``fail``。"""
        self._require_running()
        for raw in evidence_ids:
            evidence_id = str(raw).strip()
            if evidence_id and evidence_id not in self.evidence_ids:
                self.evidence_ids.append(evidence_id)
        if ok:
            if self.step_index < len(self.plan.steps) - 1:
                self.step_index += 1
            return "continue"
        self.last_error = str(message or "工具执行失败")
        if retryable and self.replan_count < self.budget.max_replans:
            self.replan_count += 1
            return "replan"
        self.fatal_tool_failure = True
        return "fail"

    def request_confirmation(self) -> None:
        self._require_running()
        self.transition(RunStatus.AWAITING_CONFIRMATION)

    def resume_after_confirmation(self) -> None:
        if self.status is not RunStatus.AWAITING_CONFIRMATION:
            raise InvalidRunTransition(
                f"只有 awaiting_confirmation 可恢复，当前为 {self.status.value}"
            )
        self.transition(RunStatus.RUNNING)

    def can_repair_verification(self) -> bool:
        return self.verification_repair_count < self.budget.max_verification_repairs

    def consume_verification_repair(self) -> int:
        if not self.can_repair_verification():
            raise AgentBudgetExceeded("verification_repair_exhausted", "引用修复次数已用尽")
        self.verification_repair_count += 1
        return self.verification_repair_count

    def finish(self) -> RunStatus:
        self._require_running()
        if self.fatal_tool_failure:
            return self.transition(RunStatus.FAILED, error=self.last_error)
        self.step_index = len(self.plan.steps)
        return self.transition(RunStatus.SUCCEEDED)

    def fail(self, message: str) -> RunStatus:
        self._require_running()
        return self.transition(RunStatus.FAILED, error=message)

    def block(self, message: str) -> RunStatus:
        self._require_running()
        return self.transition(RunStatus.BLOCKED, error=message)

    def _require_running(self) -> None:
        if self.status is not RunStatus.RUNNING:
            raise InvalidRunTransition(
                f"该操作要求 running 状态，当前为 {self.status.value}"
            )


class Executor:
    """对运行时预算的窄封装，便于 chat 与评测使用相同路径。"""

    def __init__(self, runtime: AgentRuntime, *, external_tools: Iterable[str] = ()):
        self.runtime = runtime
        self.external_tools = frozenset(str(name) for name in external_tools)

    def begin_round(self) -> int:
        return self.runtime.begin_round()

    def before_tool(self, name: str) -> None:
        self.runtime.authorize_tool(name, external=name in self.external_tools)

    def after_tool(
        self,
        *,
        ok: bool,
        retryable: bool = False,
        evidence_ids: Iterable[object] = (),
        message: Optional[str] = None,
    ) -> str:
        return self.runtime.observe_tool_result(
            ok=ok,
            retryable=retryable,
            evidence_ids=evidence_ids,
            message=message,
        )


_NUMERIC_CITATION_RE = re.compile(r"\[(\d+)\]")
_EVIDENCE_CITATION_RE = re.compile(r"\[E:([^\]\s]+)\]")
_ANY_EVIDENCE_MARKER_RE = re.compile(r"\[E:[^\]]*\]")
_ABSTENTION_RE = re.compile(
    r"^(?:根据已有资料|根据现有资料|现有资料|参考资料)?\s*(?:不足(?:以回答)?[，,:：;；]?)?"
    r"(?:无法回答|不能回答|无法确定)(?:该问题|这个问题)?[。.!！?？\s]*$"
)


@dataclass(frozen=True)
class CitationVerification:
    ok: bool
    code: str
    message: str
    citations: tuple[str, ...] = ()


class CitationVerificationError(ValueError):
    def __init__(self, result: CitationVerification):
        super().__init__(result.message)
        self.result = result
        self.code = result.code


def is_abstention(answer_text: str) -> bool:
    return bool(_ABSTENTION_RE.fullmatch(str(answer_text or "").strip()))


def verify_citations(
    answer_text: str,
    *,
    source_count: Optional[int] = None,
    evidence_ids: Optional[Iterable[object]] = None,
) -> CitationVerification:
    """验证编号引用或稳定证据引用。

    ``evidence_ids`` 非 ``None`` 时启用 Agent 模式，并只接受精确的
    ``[E:<id>]``；否则启用 ``ask`` 的 ``[n]`` 模式。
    """
    text = str(answer_text or "").strip()
    if not text:
        return CitationVerification(False, "empty_answer", "LLM 返回了空回答")

    if evidence_ids is not None:
        allowed = tuple(dict.fromkeys(str(raw).strip() for raw in evidence_ids if str(raw).strip()))
        allowed_set = frozenset(allowed)
        citations = tuple(_EVIDENCE_CITATION_RE.findall(text))
        malformed = tuple(
            marker for marker in _ANY_EVIDENCE_MARKER_RE.findall(text)
            if not _EVIDENCE_CITATION_RE.fullmatch(marker)
        )
        if malformed:
            return CitationVerification(
                False,
                "evidence_citation_malformed",
                "回答包含格式错误的证据引用；必须使用 [E:<id>]",
                citations,
            )
        unknown = tuple(dict.fromkeys(cid for cid in citations if cid not in allowed_set))
        if unknown:
            rendered = "、".join(f"[E:{cid}]" for cid in unknown)
            return CitationVerification(
                False,
                "evidence_citation_out_of_scope",
                f"LLM 返回了不在本次工具证据范围内的引用：{rendered}",
                citations,
            )
        if allowed and not citations and not is_abstention(text):
            return CitationVerification(
                False,
                "evidence_citation_missing",
                "LLM 回答缺少稳定的 [E:<id>] 证据引用",
            )
        if not allowed and citations:
            return CitationVerification(
                False,
                "evidence_citation_out_of_scope",
                "本次运行没有可引用的证据 ID",
                citations,
            )
        return CitationVerification(True, "ok", "引用验证通过", citations)

    if source_count is None:
        raise ValueError("编号引用验证必须提供 source_count")
    if not isinstance(source_count, int) or isinstance(source_count, bool) or source_count < 0:
        raise ValueError("source_count 必须是非负整数")
    numbers = tuple(int(raw) for raw in _NUMERIC_CITATION_RE.findall(text))
    invalid = tuple(dict.fromkeys(number for number in numbers if number < 1 or number > source_count))
    if invalid:
        rendered = "、".join(f"[{number}]" for number in invalid)
        return CitationVerification(
            False,
            "numeric_citation_out_of_range",
            f"LLM 返回了不存在的来源引用：{rendered}",
            tuple(str(number) for number in numbers),
        )
    if source_count and not numbers and not is_abstention(text):
        return CitationVerification(
            False,
            "numeric_citation_missing",
            "LLM 回答缺少 [n] 来源引用",
        )
    return CitationVerification(
        True, "ok", "引用验证通过", tuple(str(number) for number in numbers)
    )


def require_valid_citations(
    answer_text: str,
    *,
    source_count: Optional[int] = None,
    evidence_ids: Optional[Iterable[object]] = None,
) -> CitationVerification:
    result = verify_citations(
        answer_text, source_count=source_count, evidence_ids=evidence_ids
    )
    if not result.ok:
        raise CitationVerificationError(result)
    return result


class Verifier:
    """最终回答验证器。"""

    def verify_numeric(self, answer_text: str, source_count: int) -> CitationVerification:
        return verify_citations(answer_text, source_count=source_count)

    def verify_evidence(
        self, answer_text: str, evidence_ids: Iterable[object]
    ) -> CitationVerification:
        return verify_citations(answer_text, evidence_ids=evidence_ids)
