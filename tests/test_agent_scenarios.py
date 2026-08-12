import json
from pathlib import Path

import pytest

from paper_agent.agent_eval import (
    AgentScenario,
    SCENARIO_SCHEMA_VERSION,
    ScenarioFormatError,
    evaluate_suite,
    load_scenarios,
    run_scenario,
)
from paper_agent.agent import AgentBudget


SCENARIO_DIR = Path(__file__).parent / "scenarios"


def test_versioned_suite_has_required_coverage_and_at_least_25_scenarios():
    scenarios = load_scenarios(SCENARIO_DIR)
    assert len(scenarios) >= 25
    assert {scenario.version for scenario in scenarios} == {SCENARIO_SCHEMA_VERSION}
    categories = {scenario.category for scenario in scenarios}
    assert {
        "tool-sequence",
        "confirmation",
        "abstention",
        "failure",
        "budget",
        "citation",
        "prompt-injection",
        "cancellation",
    }.issubset(categories)
    assert sum("injection" in scenario.tags for scenario in scenarios) >= 3


def test_all_deterministic_scenarios_meet_their_expectations():
    evaluations = evaluate_suite(load_scenarios(SCENARIO_DIR))
    failures = {
        result.scenario_id: result.failures for result in evaluations if not result.passed
    }
    assert failures == {}


def test_scenario_execution_is_deterministic():
    scenario = next(
        scenario
        for scenario in load_scenarios(SCENARIO_DIR)
        if scenario.id == "citation-repaired-once"
    )
    assert run_scenario(scenario) == run_scenario(scenario)


def test_loader_rejects_unknown_version(tmp_path):
    path = tmp_path / "future.json"
    path.write_text(
        json.dumps({"version": 999, "scenarios": []}), encoding="utf-8"
    )
    with pytest.raises(ScenarioFormatError, match="不支持的场景版本"):
        load_scenarios(path)


def test_loader_rejects_duplicate_ids(tmp_path):
    scenario = {
        "id": "same",
        "category": "tool-sequence",
        "objective": "x",
        "script": [{"type": "llm", "content": "x"}],
        "expected": {"terminal_status": "succeeded", "stop_code": "completed"},
    }
    path = tmp_path / "duplicate.json"
    path.write_text(
        json.dumps({"version": 1, "scenarios": [scenario, scenario]}),
        encoding="utf-8",
    )
    with pytest.raises(ScenarioFormatError, match="重复"):
        load_scenarios(path)


@pytest.mark.parametrize(
    "expected, message",
    [
        ({"terminal_status": "succeeded"}, "expected 缺少字段"),
        (
            {
                "terminal_status": "succeeded",
                "stop_code": "completed",
                "typo_status": "succeeded",
            },
            "expected 包含未知字段",
        ),
    ],
)
def test_loader_rejects_incomplete_or_unknown_expectations(tmp_path, expected, message):
    payload = {
        "version": 1,
        "scenarios": [
            {
                "id": "invalid-expected",
                "category": "tool-sequence",
                "objective": "x",
                "script": [{"type": "llm", "content": "完成"}],
                "expected": expected,
            }
        ],
    }
    path = tmp_path / "invalid-expected.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ScenarioFormatError, match=message):
        load_scenarios(path)


def test_budget_stop_code_is_not_overwritten_by_trailing_actions():
    scenario = AgentScenario(
        id="budget-with-trailing-action",
        category="budget",
        objective="预算触发后停止",
        budget=AgentBudget(max_tool_calls=1),
        script=(
            {
                "type": "llm",
                "tool_calls": [
                    {"name": "library_status", "result": {"ok": True}},
                    {"name": "list_papers", "result": {"ok": True}},
                ],
            },
            {"type": "llm", "content": "不应执行"},
        ),
        expected={"terminal_status": "blocked", "stop_code": "tool_budget_exceeded"},
    )
    observation = run_scenario(scenario)
    assert observation.terminal_status == "blocked"
    assert observation.stop_code == "tool_budget_exceeded"


@pytest.mark.parametrize("retryable", [False, True])
def test_failed_or_replanned_tool_stops_remaining_calls_in_same_batch(retryable):
    scenario = AgentScenario(
        id=f"stop-tool-batch-{retryable}",
        category="failure",
        objective="失败后停止当前工具批次",
        script=(
            {
                "type": "llm",
                "tool_calls": [
                    {
                        "name": "local_search",
                        "result": {
                            "ok": False,
                            "code": "temporary" if retryable else "fatal",
                            "retryable": retryable,
                        },
                    },
                    {"name": "save_note", "result": {"ok": True}},
                ],
            },
        ),
        expected={
            "terminal_status": "running",
            "stop_code": "temporary" if retryable else "fatal",
        },
    )
    observation = run_scenario(scenario)
    assert observation.tool_sequence == ("local_search",)
    assert observation.replans == (1 if retryable else 0)
    assert observation.stop_code == ("temporary" if retryable else "fatal")
