from types import SimpleNamespace

import pytest

from pragent.agent import (
    AgentBudget,
    AgentBudgetExceeded,
    AgentRuntime,
    CitationVerificationError,
    Executor,
    InvalidRunTransition,
    Planner,
    RunStatus,
    is_abstention,
    require_valid_citations,
    validate_run_transition,
    verify_citations,
)
from pragent.llm import _response_metadata


def make_runtime(**budget_values):
    runtime = AgentRuntime(
        plan=Planner().create_plan("完成受控任务"),
        budget=AgentBudget(**budget_values),
    )
    runtime.start()
    return runtime


def test_run_status_values_and_legal_graph():
    assert {status.value for status in RunStatus} == {
        "proposed",
        "running",
        "awaiting_confirmation",
        "succeeded",
        "failed",
        "cancelled",
        "blocked",
    }
    assert validate_run_transition("proposed", "running") == (
        RunStatus.PROPOSED,
        RunStatus.RUNNING,
    )
    assert validate_run_transition("running", "awaiting_confirmation") == (
        RunStatus.RUNNING,
        RunStatus.AWAITING_CONFIRMATION,
    )
    assert validate_run_transition("awaiting_confirmation", "running") == (
        RunStatus.AWAITING_CONFIRMATION,
        RunStatus.RUNNING,
    )


@pytest.mark.parametrize(
    ("source", "target"),
    [
        ("proposed", "succeeded"),
        ("running", "running"),
        ("succeeded", "running"),
        ("failed", "cancelled"),
        ("cancelled", "running"),
        ("blocked", "running"),
    ],
)
def test_illegal_run_transitions_rejected(source, target):
    with pytest.raises(InvalidRunTransition):
        validate_run_transition(source, target)


def test_planner_requires_bounded_nonempty_plan():
    planner = Planner(max_steps=2)
    plan = planner.create_plan(" 查找论文 ", ["检索", "验证"])
    assert plan.objective == "查找论文"
    assert plan.steps == ("检索", "验证")
    with pytest.raises(ValueError, match="objective"):
        planner.create_plan("   ")
    with pytest.raises(ValueError, match="最多"):
        planner.create_plan("x", ["1", "2", "3"])


def test_round_and_tool_budgets_are_hard_limits():
    runtime = make_runtime(max_rounds=1, max_tool_calls=1)
    executor = Executor(runtime)
    assert executor.begin_round() == 1
    executor.before_tool("library_status")
    executor.after_tool(ok=True)
    with pytest.raises(AgentBudgetExceeded) as round_error:
        executor.begin_round()
    assert round_error.value.code == "round_budget_exceeded"
    assert runtime.status is RunStatus.FAILED

    runtime = make_runtime(max_tool_calls=1)
    executor = Executor(runtime)
    executor.before_tool("library_status")
    with pytest.raises(AgentBudgetExceeded) as tool_error:
        executor.before_tool("list_papers")
    assert tool_error.value.code == "tool_budget_exceeded"
    assert runtime.status is RunStatus.BLOCKED


def test_external_budget_and_retryable_replan_are_bounded():
    runtime = make_runtime(max_external_calls=1, max_replans=1)
    executor = Executor(runtime, external_tools={"web_search"})
    executor.before_tool("web_search")
    assert executor.after_tool(ok=False, retryable=True, message="超时") == "replan"
    assert runtime.replan_count == 1
    assert executor.after_tool(ok=False, retryable=True, message="仍超时") == "fail"
    assert runtime.fatal_tool_failure is True
    with pytest.raises(AgentBudgetExceeded, match="仍超时"):
        executor.before_tool("library_status")
    assert runtime.status is RunStatus.FAILED

    runtime = make_runtime(max_external_calls=0)
    executor = Executor(runtime, external_tools={"web_search"})
    with pytest.raises(AgentBudgetExceeded) as exc_info:
        executor.before_tool("web_search")
    assert exc_info.value.code == "external_call_budget_exceeded"
    assert runtime.external_call_count == 0


def test_confirmation_pause_resume_and_terminal_guard():
    runtime = make_runtime()
    runtime.request_confirmation()
    assert runtime.status is RunStatus.AWAITING_CONFIRMATION
    with pytest.raises(InvalidRunTransition):
        runtime.begin_round()
    runtime.resume_after_confirmation()
    runtime.finish()
    assert runtime.status is RunStatus.SUCCEEDED
    with pytest.raises(InvalidRunTransition):
        runtime.resume_after_confirmation()


def test_numeric_citation_validation_and_strict_abstention():
    assert verify_citations("结论 [1]。", source_count=1).ok
    invalid = verify_citations("结论 [2]。", source_count=1)
    assert invalid.code == "numeric_citation_out_of_range"
    missing = verify_citations("没有引用的结论。", source_count=1)
    assert missing.code == "numeric_citation_missing"
    assert is_abstention("根据已有资料无法回答。")
    assert not is_abstention("根据已有资料无法回答，但我猜结论是 A。")
    with pytest.raises(CitationVerificationError) as exc_info:
        require_valid_citations("没有引用。", source_count=1)
    assert exc_info.value.code == "numeric_citation_missing"


def test_evidence_citations_require_exact_allowed_ids():
    assert verify_citations(
        "结论 [E:local:chunk:1]。", evidence_ids=["local:chunk:1"]
    ).ok
    missing = verify_citations("结论。", evidence_ids=["local:chunk:1"])
    assert missing.code == "evidence_citation_missing"
    unknown = verify_citations(
        "结论 [E:local:chunk:2]。", evidence_ids=["local:chunk:1"]
    )
    assert unknown.code == "evidence_citation_out_of_scope"
    assert verify_citations(
        "根据已有资料无法回答。", evidence_ids=["local:chunk:1"]
    ).ok
    malformed = verify_citations("结论 [E:]。", evidence_ids=["local:chunk:1"])
    assert malformed.code == "evidence_citation_malformed"
    fabricated = verify_citations("结论 [E:fake]。", evidence_ids=[])
    assert fabricated.code == "evidence_citation_out_of_scope"


def test_response_metadata_keeps_usage_finish_reason_and_id():
    response = SimpleNamespace(
        id="resp-123",
        usage=SimpleNamespace(
            prompt_tokens=10, completion_tokens=4, total_tokens=14
        ),
        choices=[SimpleNamespace(finish_reason="tool_calls")],
    )
    assert _response_metadata(response) == {
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 4,
            "total_tokens": 14,
        },
        "finish_reason": "tool_calls",
        "response_id": "resp-123",
    }
