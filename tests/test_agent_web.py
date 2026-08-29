"""Web Agent：SSE 受控对话、确认/取消与 run 审计端点。"""
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient as FastAPITestClient

from pragent.store import Store
from pragent.storage.research_repository import ResearchRepository
from pragent.tool_protocol import ToolEffect, ToolResult, ToolSpec
from pragent.tools import register_tool, unregister_tool
from pragent.webapp import create_app

from helpers import FakeEmbedder, StreamingScriptLLM


def TestClient(app, **kwargs):
    kwargs.setdefault("base_url", "http://127.0.0.1")
    return FastAPITestClient(app, **kwargs)


_EXECUTED: list[str] = []
_TERMINAL_RUN_STATUSES = {"succeeded", "failed", "cancelled", "blocked"}


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
    assert llm.calls[0][-1] == {"role": "user", "content": "库里有什么？"}
    restored = store.load_agent_messages("s1")
    assert [message["role"] for message in restored] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert restored[0]["content"] == "库里有什么？"
    assert restored[1]["tool_calls"][0]["id"] == "c1"
    assert restored[2]["tool_call_id"] == "c1"
    assert restored[3]["content"] == "库里有 1 篇论文。"


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
    pending_state = store.load_agent_session_state("s2")
    assert [message["role"] for message in pending_state["messages"]] == ["user", "assistant"]
    assert pending_state["pending_action"]["tool_call_id"] == "c2"

    r2 = client.post("/api/agent/confirm", json={"session_id": "s2", "confirm": True})
    assert r2.status_code == 200
    assert '"code": "confirmed"' in r2.text
    assert "写好了" in r2.text
    assert '"status": "succeeded"' in r2.text
    assert _EXECUTED == ["hello"]
    assert store.list_agent_runs()[0].status == "succeeded"
    restored = store.load_agent_messages("s2")
    assert [message["role"] for message in restored] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert restored[2]["tool_call_id"] == "c2"


def test_pending_confirmation_survives_restart_and_history_is_redrawn(tmp_path, fake_write):
    db_path = tmp_path / "restart-pending.db"
    first_store = Store(db_path)
    first_llm = StreamingScriptLLM(
        [{
            "content": "需要确认。",
            "tool_calls": [{"id": "restart-call", "name": "fake_write", "arguments": {"text": "once"}}],
        }]
    )
    first_client = TestClient(
        create_app(store=first_store, embedder=FakeEmbedder(), llm=first_llm)
    )
    response = first_client.post(
        "/api/agent/chat", json={"session_id": "restart-session", "question": "执行写入"}
    )
    assert response.status_code == 200
    state = first_client.get("/api/agent/sessions/restart-session").json()
    assert [item["role"] for item in state["history"] if item["type"] == "message"] == [
        "user", "assistant"
    ]
    assert state["pending"]["name"] == "fake_write"

    restarted_store = Store(db_path)
    restarted_llm = StreamingScriptLLM(
        [{"content": "重启后完成。", "tool_calls": [], "deltas": ["重启后完成。"]}]
    )
    restarted_client = TestClient(
        create_app(store=restarted_store, embedder=FakeEmbedder(), llm=restarted_llm)
    )
    confirmed = restarted_client.post(
        "/api/agent/confirm", json={"session_id": "restart-session", "confirm": True}
    )
    assert confirmed.status_code == 200
    assert "重启后完成" in confirmed.text
    assert _EXECUTED == ["once"]
    assert restarted_store.load_agent_session_state("restart-session")["pending_action"] is None


def test_agent_session_binds_project_and_rejects_scope_switch(tmp_path):
    db_path = tmp_path / "project-session.db"
    store = Store(db_path)
    repository = ResearchRepository(db_path)
    first_project = repository.create_project("项目一")
    second_project = repository.create_project("项目二")
    llm = StreamingScriptLLM([{"content": "已绑定。", "tool_calls": []}])
    client = TestClient(create_app(store=store, embedder=FakeEmbedder(), llm=llm))

    response = client.post(
        "/api/agent/chat",
        json={
            "session_id": "scoped",
            "project_id": first_project.id,
            "question": "读取当前项目",
        },
    )
    assert response.status_code == 200
    state = client.get("/api/agent/sessions/scoped").json()
    assert state["project_id"] == first_project.id
    run = store.list_agent_runs(session_id="scoped")[0]
    assert run.project_id == first_project.id
    assert run.session_id == "scoped"

    mismatch = client.post(
        "/api/agent/chat",
        json={
            "session_id": "scoped",
            "project_id": second_project.id,
            "question": "切换项目",
        },
    )
    assert mismatch.status_code == 409


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
    restored = store.load_agent_messages("s3")
    assert [message["role"] for message in restored] == ["user", "assistant", "tool"]
    assert restored[-1]["tool_call_id"] == "c3"


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


def test_agent_session_restores_after_new_app_and_keeps_sessions_isolated(tmp_path):
    db_path = tmp_path / "t.db"
    first_store = Store(db_path)
    first_llm = StreamingScriptLLM(
        [
            {"content": "A 的首轮", "tool_calls": []},
            {"content": "B 的首轮", "tool_calls": []},
        ]
    )
    first_client = TestClient(
        create_app(store=first_store, embedder=FakeEmbedder(), llm=first_llm)
    )
    assert first_client.post(
        "/api/agent/chat", json={"session_id": "A", "question": "问题 A1"}
    ).status_code == 200
    assert first_client.post(
        "/api/agent/chat", json={"session_id": "B", "question": "问题 B1"}
    ).status_code == 200

    restarted_store = Store(db_path)
    restarted_llm = StreamingScriptLLM(
        [
            {"content": "A 的续答", "tool_calls": []},
            {"content": "B 的续答", "tool_calls": []},
        ]
    )
    restarted_client = TestClient(
        create_app(
            store=restarted_store,
            embedder=FakeEmbedder(),
            llm=restarted_llm,
        )
    )
    assert restarted_client.post(
        "/api/agent/chat", json={"session_id": "A", "question": "问题 A2"}
    ).status_code == 200
    assert restarted_client.post(
        "/api/agent/chat", json={"session_id": "B", "question": "问题 B2"}
    ).status_code == 200

    assert [message.get("content") for message in restarted_llm.calls[0]] == [
        "问题 A1",
        "A 的首轮",
        "问题 A2",
    ]
    assert [message.get("content") for message in restarted_llm.calls[1]] == [
        "问题 B1",
        "B 的首轮",
        "问题 B2",
    ]
    first_store.close()
    restarted_store.close()


def test_agent_clear_session_is_idempotent_and_prevents_restore(tmp_path):
    store = Store(tmp_path / "t.db")
    llm = StreamingScriptLLM(
        [
            {"content": "旧回答", "tool_calls": []},
            {"content": "新回答", "tool_calls": []},
        ]
    )
    client = TestClient(create_app(store=store, embedder=FakeEmbedder(), llm=llm))
    client.post("/api/agent/chat", json={"session_id": "clear-me", "question": "旧问题"})

    cleared = client.delete("/api/agent/sessions/clear-me")
    assert cleared.status_code == 200
    assert cleared.json()["cleared"] is True
    repeated = client.delete("/api/agent/sessions/clear-me")
    assert repeated.status_code == 200
    assert repeated.json()["cleared"] is False

    client.post("/api/agent/chat", json={"session_id": "clear-me", "question": "新问题"})
    assert [message.get("content") for message in llm.calls[-1]] == ["新问题"]


def test_agent_clear_rejects_pending_session(tmp_path, fake_write):
    llm = StreamingScriptLLM(
        [
            {
                "content": None,
                "tool_calls": [
                    {"id": "pending-clear", "name": "fake_write", "arguments": {"text": "x"}}
                ],
            }
        ]
    )
    _, client = _app(tmp_path, llm)
    client.post("/api/agent/chat", json={"session_id": "pending", "question": "写入"})

    response = client.delete("/api/agent/sessions/pending")
    assert response.status_code == 409
    assert "待确认" in response.json()["detail"]


def test_agent_rejects_concurrent_turn_and_clear_for_same_session(tmp_path, monkeypatch):
    import pragent.agent_api as agent_api

    started = threading.Event()
    release = threading.Event()

    def blocking_run_turn(session, emit, *, objective=None, scope=None):
        started.set()
        release.wait(timeout=5)
        # worker 结束必须先于流终止哨兵（scope.worker_finished 内部发送）。
        if scope is not None:
            scope.worker_finished()
        else:
            emit(None)

    monkeypatch.setattr(agent_api, "_run_turn", blocking_run_turn)
    app = create_app(
        store=Store(tmp_path / "t.db"),
        embedder=FakeEmbedder(),
        llm=StreamingScriptLLM([]),
    )
    first_client = TestClient(app)
    second_client = TestClient(app)
    with ThreadPoolExecutor(max_workers=1) as executor:
        first = executor.submit(
            first_client.post,
            "/api/agent/chat",
            json={"session_id": "busy", "question": "第一问"},
        )
        assert started.wait(timeout=2)
        concurrent = second_client.post(
            "/api/agent/chat", json={"session_id": "busy", "question": "第二问"}
        )
        clearing = second_client.delete("/api/agent/sessions/busy")
        release.set()
        assert first.result(timeout=5).status_code == 200

    assert concurrent.status_code == 409
    assert clearing.status_code == 409


# ---------- Step 26：SSE 断开竞态、cancel event 与迟到事件合同 ----------


def test_turn_scope_releases_lock_only_after_worker_finishes():
    """断开竞态合同：流退出不得释放未完成 worker 的排他锁。"""
    import pragent.agent_api as agent_api

    session = agent_api.WebAgentSession("scope-test", store=None, embedder=None, llm=None)
    assert session.lock.acquire(blocking=False)
    scope = agent_api._TurnScope(session)
    scope.bind_emitter(lambda event: None)

    # 客户端断开：取消置位，但 worker 未结束，锁必须保持占用。
    scope.stream_exited()
    assert scope.cancel_event.is_set()
    assert session.lock.locked()

    # worker 真正结束后释放；重复调用不产生第二次 release。
    scope.worker_finished()
    assert not session.lock.locked()
    scope.worker_finished()
    assert not session.lock.locked()


def test_late_events_are_dropped_after_stream_exit():
    """迟到事件合同：流退出后 emit 直接丢弃，不再进入队列。"""
    import asyncio

    import pragent.agent_api as agent_api

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    try:
        session = agent_api.WebAgentSession("late", store=None, embedder=None, llm=None)
        scope = agent_api._TurnScope(session)
        aq = asyncio.Queue()
        emit = agent_api._sse_emitter(loop, aq, scope)

        emit({"type": "session", "session_id": "late"})
        scope.stream_exited()  # 模拟客户端断开
        emit({"type": "assistant_delta", "text": "迟到事件"})
        emit(None)

        async def drain():
            items = []
            while not aq.empty():
                items.append(await aq.get())
            return items

        events = asyncio.run_coroutine_threadsafe(drain(), loop).result(timeout=2)
        assert events == [{"type": "session", "session_id": "late"}]
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=2)
        loop.close()


def test_disconnect_mid_turn_cancels_run_and_releases_lock(tmp_path):
    """断开回合在阶段边界终止：run=cancelled、消息回到上一持久化边界。"""
    import time as time_mod

    import pragent.agent_api as agent_api

    store = Store(tmp_path / "disc.db")
    started = threading.Event()
    release = threading.Event()

    class BlockingLLM:
        is_configured = True
        supports_streaming = False

        def chat_with_tools(self, system, messages, tools):
            started.set()
            release.wait(timeout=5)
            return {
                "content": None,
                "tool_calls": [{"id": "c1", "name": "list_papers", "arguments": {}}],
            }

    session = agent_api.WebAgentSession("disc", store, FakeEmbedder(), BlockingLLM())
    store.ensure_agent_session("disc")  # 真实流程中由 registry 完成
    assert session.lock.acquire(blocking=False)
    scope = agent_api._TurnScope(session)
    events = []
    scope.bind_emitter(events.append)

    agent_api._run_turn(session, events.append, objective="断开测试", scope=scope)
    assert started.wait(timeout=2)
    scope.stream_exited()  # 模拟 SSE 断开
    assert scope.cancel_event.is_set()
    assert session.lock.locked()  # worker 未结束，锁不得释放
    release.set()

    deadline = time_mod.time() + 5
    while time_mod.time() < deadline and session.lock.locked():
        time_mod.sleep(0.02)
    assert not session.lock.locked()

    runs = store.list_agent_runs()
    assert len(runs) == 1
    assert runs[0].status == "cancelled"
    state = store.load_agent_session_state("disc")
    # 半截回合不进入持久化边界；pending 不存在。
    assert state["messages"] == []
    assert state["pending_action"] is None
    # 终态 complete 事件已生成（流断开时由 Store 记录兜底）。
    assert any(
        event.get("type") == "complete" and event.get("status") == "cancelled"
        for event in events
    )


def test_confirm_disconnect_keeps_executed_tool_result(tmp_path, fake_write):
    """确认续跑中断开：已执行副作用工具的 tool result 必须保留为终态。"""
    import time as time_mod

    import pragent.agent_api as agent_api

    store = Store(tmp_path / "confirm-disc.db")
    resume_started = threading.Event()
    release = threading.Event()

    class PhaseLLM:
        is_configured = True
        supports_streaming = False

        def __init__(self):
            self.calls = 0

        def chat_with_tools(self, system, messages, tools):
            self.calls += 1
            if self.calls == 1:
                return {
                    "content": "需要写入。",
                    "tool_calls": [
                        {"id": "cw", "name": "fake_write", "arguments": {"text": "keep"}}
                    ],
                }
            resume_started.set()
            release.wait(timeout=5)
            # 返回工具调用迫使进入下一轮，从而命中阶段边界取消检查。
            return {
                "content": None,
                "tool_calls": [{"id": "c2", "name": "list_papers", "arguments": {}}],
            }

    session = agent_api.WebAgentSession("cd", store, FakeEmbedder(), PhaseLLM())
    store.ensure_agent_session("cd")  # 真实流程中由 registry 完成
    events = []

    # 回合 1：产生待确认票据并完整停在 awaiting_confirmation 边界。
    first_scope = agent_api._TurnScope(session)
    first_scope.bind_emitter(events.append)
    agent_api._run_turn(session, events.append, objective="写入", scope=first_scope)

    def saw_awaiting_complete():
        return any(
            isinstance(event, dict)
            and event.get("type") == "complete"
            and event.get("status") == "awaiting_confirmation"
            for event in events
        )

    deadline = time_mod.time() + 5
    while time_mod.time() < deadline and not saw_awaiting_complete():
        time_mod.sleep(0.02)
    assert saw_awaiting_complete()
    pending = session.ctx.pending_action
    assert pending is not None and pending.name == "fake_write"

    # 回合 2：确认续跑，在恢复的 LLM 调用阻塞期间模拟客户端断开。
    second_scope = agent_api._TurnScope(session)
    second_scope.bind_emitter(events.append)
    agent_api._run_confirmation(
        session, events.append, pending=pending, confirm=True, scope=second_scope
    )
    assert resume_started.wait(timeout=2)
    second_scope.stream_exited()
    assert second_scope.cancel_event.is_set()
    release.set()

    # 等待 run 终态与收尾 complete 事件（persist 在 finalize 之前完成）。
    def terminal_complete():
        runs = store.list_agent_runs()
        done = bool(runs) and runs[0].status in _TERMINAL_RUN_STATUSES
        complete = any(
            isinstance(event, dict)
            and event.get("type") == "complete"
            and event.get("status") == "cancelled"
            for event in events
        )
        return done and complete

    deadline = time_mod.time() + 5
    while time_mod.time() < deadline and not terminal_complete():
        time_mod.sleep(0.02)
    assert terminal_complete()

    runs = store.list_agent_runs()
    assert len(runs) == 1
    assert runs[0].status == "cancelled"
    state = store.load_agent_session_state("cd")
    roles = [message["role"] for message in state["messages"]]
    # 协议闭合、已执行工具的回执保留，绝不回滚成"从未发生"。
    assert roles == ["user", "assistant", "tool", "assistant", "tool"]
    assert state["messages"][2]["tool_call_id"] == "cw"
    assert state["pending_action"] is None
    assert _EXECUTED == ["keep"]
