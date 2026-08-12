"""版本化、无网络、确定性的 Agent 场景评测框架。"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from .agent import (
    AgentBudget,
    AgentBudgetExceeded,
    AgentRuntime,
    Executor,
    InvalidRunTransition,
    Planner,
    RunStatus,
    Verifier,
)

SCENARIO_SCHEMA_VERSION = 1
_NETWORK_TOOLS = frozenset({"web_search", "download_paper"})
_EXPECTED_FIELDS = frozenset(
    {
        "terminal_status",
        "tool_sequence",
        "confirmation_requested",
        "stop_code",
        "verification_codes",
        "max_rounds",
        "max_external_calls",
        "forbidden_tools",
    }
)
_REQUIRED_EXPECTED_FIELDS = frozenset({"terminal_status", "stop_code"})


class ScenarioFormatError(ValueError):
    pass


@dataclass(frozen=True)
class AgentScenario:
    id: str
    category: str
    objective: str
    script: tuple[dict[str, Any], ...]
    expected: dict[str, Any]
    budget: AgentBudget = field(default_factory=AgentBudget)
    tags: tuple[str, ...] = ()
    version: int = SCENARIO_SCHEMA_VERSION


@dataclass(frozen=True)
class ScenarioObservation:
    terminal_status: str
    tool_sequence: tuple[str, ...]
    confirmation_requested: bool
    rounds: int
    external_calls: int
    replans: int
    verification_codes: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    stop_code: str


@dataclass(frozen=True)
class ScenarioEvaluation:
    scenario_id: str
    passed: bool
    failures: tuple[str, ...]
    observation: ScenarioObservation


def load_scenarios(path: str | Path) -> list[AgentScenario]:
    """读取一个 JSON 文件或目录内全部 ``*.json`` 场景。"""
    source = Path(path)
    files = sorted(source.glob("*.json")) if source.is_dir() else [source]
    scenarios: list[AgentScenario] = []
    seen: set[str] = set()
    for file_path in files:
        try:
            payload = json.loads(file_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ScenarioFormatError(f"无法读取场景文件 {file_path}: {exc}") from exc
        if not isinstance(payload, Mapping):
            raise ScenarioFormatError(f"场景文件必须是 JSON 对象：{file_path}")
        version = payload.get("version")
        if version != SCENARIO_SCHEMA_VERSION:
            raise ScenarioFormatError(
                f"不支持的场景版本 {version!r}；当前为 {SCENARIO_SCHEMA_VERSION}"
            )
        raw_scenarios = payload.get("scenarios")
        if not isinstance(raw_scenarios, list):
            raise ScenarioFormatError(f"scenarios 必须是数组：{file_path}")
        for raw in raw_scenarios:
            scenario = _parse_scenario(raw, version)
            if scenario.id in seen:
                raise ScenarioFormatError(f"场景 id 重复：{scenario.id}")
            seen.add(scenario.id)
            scenarios.append(scenario)
    return scenarios


def run_scenario(scenario: AgentScenario) -> ScenarioObservation:
    """解释脚本并通过真实 AgentRuntime 边界产生确定性观察值。"""
    runtime = AgentRuntime(
        plan=Planner().create_plan(scenario.objective), budget=scenario.budget
    )
    runtime.start()
    executor = Executor(runtime, external_tools=_NETWORK_TOOLS)
    tool_sequence: list[str] = []
    verification_codes: list[str] = []
    confirmation_requested = False
    stop_code = "script_exhausted"

    for action in scenario.script:
        action_type = action["type"]
        if runtime.is_terminal:
            break

        if action_type == "cancel":
            runtime.transition(RunStatus.CANCELLED)
            stop_code = "cancelled"
            break

        if action_type == "confirm":
            if runtime.status is not RunStatus.AWAITING_CONFIRMATION:
                raise ScenarioFormatError(
                    f"{scenario.id}: confirm 只能跟在 confirmation_required 后"
                )
            if action.get("decision", "approve") != "approve":
                runtime.transition(RunStatus.CANCELLED)
                stop_code = "confirmation_denied"
                break
            runtime.resume_after_confirmation()
            result = _result_mapping(action.get("result", {"ok": True, "code": "ok"}))
            executor.after_tool(
                ok=result["ok"],
                retryable=result["retryable"],
                evidence_ids=result["evidence_ids"],
                message=result["message"],
            )
            stop_code = result["code"]
            continue

        if action_type != "llm":
            raise ScenarioFormatError(f"{scenario.id}: 未知脚本动作 {action_type!r}")
        if runtime.status is RunStatus.AWAITING_CONFIRMATION:
            stop_code = "confirmation_not_resolved"
            break
        try:
            executor.begin_round()
        except AgentBudgetExceeded as exc:
            stop_code = exc.code
            break

        tool_calls = action.get("tool_calls", [])
        if tool_calls:
            for raw_call in tool_calls:
                name = str(raw_call["name"])
                tool_sequence.append(name)
                try:
                    executor.before_tool(name)
                except AgentBudgetExceeded as exc:
                    stop_code = exc.code
                    break
                result = _result_mapping(raw_call.get("result", {"ok": True, "code": "ok"}))
                stop_code = result["code"]
                if result["code"] == "confirmation_required":
                    runtime.request_confirmation()
                    confirmation_requested = True
                    break
                disposition = executor.after_tool(
                    ok=result["ok"],
                    retryable=result["retryable"],
                    evidence_ids=result["evidence_ids"],
                    message=result["message"],
                )
                if disposition in {"fail", "replan"}:
                    break
            continue

        content = str(action.get("content", ""))
        verification = Verifier().verify_evidence(content, runtime.evidence_ids)
        verification_codes.append(verification.code)
        stop_code = verification.code
        if not verification.ok:
            if action.get("allow_repair", False) and runtime.can_repair_verification():
                runtime.consume_verification_repair()
                continue
            runtime.fail(verification.message)
            break
        runtime.finish()
        stop_code = "completed" if runtime.status is RunStatus.SUCCEEDED else "tool_failure"
        break

    return ScenarioObservation(
        terminal_status=runtime.status.value,
        tool_sequence=tuple(tool_sequence),
        confirmation_requested=confirmation_requested,
        rounds=runtime.round_count,
        external_calls=runtime.external_call_count,
        replans=runtime.replan_count,
        verification_codes=tuple(verification_codes),
        evidence_ids=tuple(runtime.evidence_ids),
        stop_code=stop_code,
    )


def evaluate_scenario(scenario: AgentScenario) -> ScenarioEvaluation:
    observation = run_scenario(scenario)
    expected = scenario.expected
    failures: list[str] = []

    _expect_equal(failures, "terminal_status", observation.terminal_status, expected)
    _expect_equal(
        failures, "tool_sequence", list(observation.tool_sequence), expected
    )
    _expect_equal(
        failures,
        "confirmation_requested",
        observation.confirmation_requested,
        expected,
    )
    _expect_equal(failures, "stop_code", observation.stop_code, expected)
    _expect_equal(
        failures,
        "verification_codes",
        list(observation.verification_codes),
        expected,
    )
    if "max_rounds" in expected and observation.rounds > int(expected["max_rounds"]):
        failures.append(
            f"rounds={observation.rounds} 超过 max_rounds={expected['max_rounds']}"
        )
    if "max_external_calls" in expected and observation.external_calls > int(
        expected["max_external_calls"]
    ):
        failures.append(
            "external_calls="
            f"{observation.external_calls} 超过 max_external_calls={expected['max_external_calls']}"
        )
    forbidden = frozenset(str(name) for name in expected.get("forbidden_tools", []))
    called_forbidden = sorted(forbidden.intersection(observation.tool_sequence))
    if called_forbidden:
        failures.append(f"调用了禁止工具：{', '.join(called_forbidden)}")
    return ScenarioEvaluation(
        scenario_id=scenario.id,
        passed=not failures,
        failures=tuple(failures),
        observation=observation,
    )


def evaluate_suite(scenarios: Iterable[AgentScenario]) -> list[ScenarioEvaluation]:
    return [evaluate_scenario(scenario) for scenario in scenarios]


def _parse_scenario(raw: Any, version: int) -> AgentScenario:
    if not isinstance(raw, Mapping):
        raise ScenarioFormatError("每个场景必须是 JSON 对象")
    required = ("id", "category", "objective", "script", "expected")
    missing = [name for name in required if name not in raw]
    if missing:
        raise ScenarioFormatError(f"场景缺少字段：{', '.join(missing)}")
    scenario_id = str(raw["id"]).strip()
    if not scenario_id:
        raise ScenarioFormatError("场景 id 不能为空")
    script = raw["script"]
    expected = raw["expected"]
    if not isinstance(script, list) or not script:
        raise ScenarioFormatError(f"{scenario_id}: script 必须是非空数组")
    if not isinstance(expected, Mapping):
        raise ScenarioFormatError(f"{scenario_id}: expected 必须是对象")
    unknown_expected = sorted(set(expected) - _EXPECTED_FIELDS)
    if unknown_expected:
        raise ScenarioFormatError(
            f"{scenario_id}: expected 包含未知字段：{', '.join(unknown_expected)}"
        )
    missing_expected = sorted(_REQUIRED_EXPECTED_FIELDS - set(expected))
    if missing_expected:
        raise ScenarioFormatError(
            f"{scenario_id}: expected 缺少字段：{', '.join(missing_expected)}"
        )
    parsed_script = tuple(_parse_action(scenario_id, action) for action in script)
    tags = raw.get("tags", [])
    if not isinstance(tags, list):
        raise ScenarioFormatError(f"{scenario_id}: tags 必须是数组")
    try:
        budget = AgentBudget.from_value(raw.get("budget"))
    except (TypeError, ValueError) as exc:
        raise ScenarioFormatError(f"{scenario_id}: 非法 budget: {exc}") from exc
    return AgentScenario(
        id=scenario_id,
        category=str(raw["category"]),
        objective=str(raw["objective"]),
        script=parsed_script,
        expected=dict(expected),
        budget=budget,
        tags=tuple(str(tag) for tag in tags),
        version=version,
    )


def _parse_action(scenario_id: str, raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ScenarioFormatError(f"{scenario_id}: 脚本动作必须是对象")
    action = dict(raw)
    action_type = action.get("type")
    if action_type not in {"llm", "confirm", "cancel"}:
        raise ScenarioFormatError(f"{scenario_id}: 非法动作类型 {action_type!r}")
    if action_type == "llm":
        calls = action.get("tool_calls", [])
        if not isinstance(calls, list):
            raise ScenarioFormatError(f"{scenario_id}: tool_calls 必须是数组")
        for call in calls:
            if not isinstance(call, Mapping) or not str(call.get("name", "")).strip():
                raise ScenarioFormatError(f"{scenario_id}: tool_call 缺少 name")
    return action


def _result_mapping(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ScenarioFormatError("工具 result 必须是对象")
    evidence = raw.get("evidence_ids", [])
    if isinstance(evidence, str):
        evidence = [evidence]
    if not isinstance(evidence, list):
        raise ScenarioFormatError("result.evidence_ids 必须是数组")
    return {
        "ok": bool(raw.get("ok", False)),
        "code": str(raw.get("code", "ok" if raw.get("ok") else "tool_error")),
        "message": str(raw.get("message", "")),
        "retryable": bool(raw.get("retryable", False)),
        "evidence_ids": [str(item) for item in evidence],
    }


def _expect_equal(
    failures: list[str], name: str, actual: Any, expected: Mapping[str, Any]
) -> None:
    if name in expected and actual != expected[name]:
        failures.append(f"{name}: 期望 {expected[name]!r}，实际 {actual!r}")
