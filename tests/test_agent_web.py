"""Web Agent：SSE 受控对话、确认/取消与 run 审计端点。"""
import pytest
from fastapi.testclient import TestClient as FastAPITestClient

from pragent.store import Store
from pragent.tool_protocol import ToolEffect, ToolResult, ToolSpec
from pragent.tools import register_tool, unregister_tool
from pragent.webapp import create_app

from helpers import FakeEmbedder, StreamingScriptLLM


def TestClient(app, **kwargs):
    kwargs.setdefault("base_url", "http://127.0.0.1")
    return FastAPITestClient(app, **kwargs)


_EXECUTED: list[str] = []


@pytest.fixture
def fake_write():
    """注册一个仅测试用的 WRITE_LOCAL 工具（确认票据链路离线可测）。"""
    _EXECUTED.clear()

    def handler(ctx, text=""):
        _EXECUTED.append(text)
        return ToolResult.success(message=f"已写入 {len(text)} 字符")

    register_tool(
        ToolSpec(
            name="fake_write",
            description="测试写入工具",
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
            handler=handler,
            effects=frozenset({ToolEffect.WRITE_LOCAL}),
            timeout_seconds=5.0,
            idempotent=True,
        )
    )
    yield
    unregister_tool("fake_write")


def _app(tmp_path, llm):
    store = Store(tmp_path / "t.db")
    return store, TestClient(create_app(store=store, embedder=FakeEmbedder(), llm=llm))


def test_agent_chat_streams_tool_and_answer(tmp_path):
    llm = StreamingScriptLLM(
        [
            {
                "content": "先查库。",
                "tool_calls": [{"id": "c1", "name": "library_status", "arguments": {}}],
            },
            {
                "content": "库里有 1 篇论文。",
                "tool_calls": [],
                "deltas": ["库里", "有 1 篇论文。"],
            },
        ]
    )
    store, client = _app(tmp_path, llm)

    r = client.post("/api/agent/chat", json={"session_id": "s1", "question": "库里有什么？"})

    assert r.status_code == 200
    assert "text/event-stream" in r.headers["content-type"]
    body = r.text
    assert '"type": "session"' in body
    assert '"type": "tool"' in body and "library_status" in body
    assert '"type": "assistant_delta"' in body
    assert "库里" in body and "有 1 篇论文。" in body
    assert '"type": "complete"' in body and '"status": "succeeded"' in body
    runs = store.list_agent_runs()
    assert len(runs) == 1
    assert runs[0].status == "succeeded"
    assert runs[0].objective == "库里有什么？"


def test_agent_confirmation_and_resume(tmp_path, fake_write):
    llm = StreamingScriptLLM(
        [
            {
                "content": "需要写文件。",
                "tool_calls": [{"id": "c2", "name": "fake_write", "arguments": {"text": "hello"}}],
            },
            {"content": "写好了。", "tool_calls": [], "deltas": ["写好了"]},
        ]
    )
    store, client = _app(tmp_path, llm)

    r1 = client.post("/api/agent/chat", json={"session_id": "s2", "question": "写个文件"})
    assert r1.status_code == 200
    assert '"type": "pending"' in r1.text
    assert "fake_write" in r1.text
    assert '"status": "awaiting_confirmation"' in r1.text
    assert _EXECUTED == []  # 未确认前不执行

    r2 = client.post("/api/agent/confirm", json={"session_id": "s2", "confirm": True})
    assert r2.status_code == 200
    assert '"code": "confirmed"' in r2.text
    assert "写好了" in r2.text
    assert '"status": "succeeded"' in r2.text
    assert _EXECUTED == ["hello"]
    assert store.list_agent_runs()[0].status == "succeeded"


def test_agent_cancel_flow(tmp_path, fake_write):
    llm = StreamingScriptLLM(
        [
            {
                "content": None,
                "tool_calls": [{"id": "c3", "name": "fake_write", "arguments": {"text": "never"}}],
            },
        ]
    )
    store, client = _app(tmp_path, llm)

    r1 = client.post("/api/agent/chat", json={"session_id": "s3", "question": "写文件"})
    assert '"type": "pending"' in r1.text

    r2 = client.post("/api/agent/confirm", json={"session_id": "s3", "confirm": False})
    assert r2.status_code == 200
    assert '"status": "cancelled"' in r2.text
    assert _EXECUTED == []
    assert store.list_agent_runs()[0].status == "cancelled"


def test_agent_runs_and_events_audit(tmp_path):
    llm = StreamingScriptLLM(
        [{"content": "完成了。", "tool_calls": [], "deltas": ["完成了"]}]
    )
    store, client = _app(tmp_path, llm)

    client.post("/api/agent/chat", json={"session_id": "s4", "question": "目标A"})

    r = client.get("/api/agent/runs")
    assert r.status_code == 200
    items = r.json()["items"]
    assert len(items) == 1
    assert items[0]["status"] == "succeeded"
    assert items[0]["objective"] == "目标A"

    rid = items[0]["run_id"]
    r2 = client.get(f"/api/agent/runs/{rid}/events")
    assert r2.status_code == 200
    data = r2.json()
    assert data["run"]["run_id"] == rid
    kinds = [e["event_type"] for e in data["items"]]
    assert "run_created" in kinds
    assert "llm_request" in kinds
    assert "status_transition" in kinds


def test_agent_chat_validates_question(tmp_path):
    _, client = _app(tmp_path, StreamingScriptLLM([]))

    r = client.post("/api/agent/chat", json={"session_id": "s5", "question": "  "})

    assert r.status_code == 400


def test_agent_confirm_requires_pending(tmp_path):
    llm = StreamingScriptLLM(
        [{"content": "没有待确认。", "tool_calls": [], "deltas": ["没有待确认"]}]
    )
    _, client = _app(tmp_path, llm)
    client.post("/api/agent/chat", json={"session_id": "s6", "question": "随便聊聊"})

    r = client.post("/api/agent/confirm", json={"session_id": "s6", "confirm": True})
    assert r.status_code == 400

    r2 = client.post("/api/agent/confirm", json={"session_id": "unknown", "confirm": True})
    assert r2.status_code == 404


def test_agent_chat_rejects_new_turn_while_pending(tmp_path, fake_write):
    llm = StreamingScriptLLM(
        [
            {
                "content": None,
                "tool_calls": [{"id": "c4", "name": "fake_write", "arguments": {"text": "x"}}],
            },
        ]
    )
    _, client = _app(tmp_path, llm)

    r1 = client.post("/api/agent/chat", json={"session_id": "s7", "question": "写文件"})
    assert '"type": "pending"' in r1.text

    r2 = client.post("/api/agent/chat", json={"session_id": "s7", "question": "再问一个"})
    assert r2.status_code == 409
