"""Web Agent：SSE 流式受控对话、确认/取消与 run 审计。

会话与 TUI 相同的内存模型（消息历史 + ToolContext），run 与结构化事件
由 Store 持久化。浏览器通过 SSE 事件流实时看到工具调用、待确认票据与
流式回答。事件格式：

- ``{"type": "session", "session_id": ...}``
- ``{"type": "assistant_delta", "text": ...}``（模型内容增量）
- ``{"type": "tool", "name": ..., "args": {...}, "result": ..., "code": ...}``
- ``{"type": "verification", "message": ..., "code": ...}``
- ``{"type": "error", "message": ..., "code": ...}``
- ``{"type": "pending", "name": ..., "summary": ..., "digest": ..., "action_id": ...}``
- ``{"type": "run", "run_id": ...}``
- ``{"type": "complete", "status": ..., "error": ...}``（终态或等待确认）

会话为进程内内存态（上限 64 个，最久未使用先淘汰）；刷新页面或重启
服务会丢失消息历史，但 run 与事件仍可通过审计侧栏查询。
"""
import asyncio
import json
import threading
import time
from collections import OrderedDict
from typing import Any, Callable, Optional

from fastapi import HTTPException, Query
from fastapi.responses import StreamingResponse

from .chat import cancel_pending_run, chat_turn
from .tools import ToolContext, confirm_pending_action, pending_action_description

_SESSION_CAP = 64
_MAX_QUESTION_CHARS = 20_000
_MAX_SESSION_ID_CHARS = 128
_MAX_TOOL_RESULT_CHARS = 500

_SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


class WebAgentSession:
    def __init__(self, session_id: str, store, embedder, llm):
        self.id = session_id
        self.messages: list[dict] = []
        self.ctx = ToolContext(store=store, embedder=embedder, llm=llm)
        self.lock = threading.Lock()
        self.last_active = time.time()
        self.run_id: Optional[str] = None


class _SessionRegistry:
    """会话注册表；store/embedder/llm 单例惰性创建。"""

    def __init__(self, store_factory, embedder_factory, llm_factory):
        self._store_factory = store_factory
        self._embedder_factory = embedder_factory
        self._llm_factory = llm_factory
        self._store = None
        self._embedder = None
        self._llm = None
        self._sessions: OrderedDict[str, WebAgentSession] = OrderedDict()
        self._lock = threading.Lock()

    @property
    def store(self):
        if self._store is None:
            self._store = self._store_factory()
        return self._store

    def _dependencies(self):
        if self._embedder is None or self._llm is None:
            self._embedder = self._embedder_factory()
            self._llm = self._llm_factory()
        return self._embedder, self._llm

    def get(self, session_id: str) -> WebAgentSession:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                embedder, llm = self._dependencies()
                session = WebAgentSession(session_id, self.store, embedder, llm)
                self._sessions[session_id] = session
                while len(self._sessions) > _SESSION_CAP:
                    self._sessions.popitem(last=False)
            else:
                self._sessions.move_to_end(session_id)
            session.last_active = time.time()
            return session

    def find(self, session_id: str) -> Optional[WebAgentSession]:
        with self._lock:
            return self._sessions.get(session_id)


def _event_json(event: dict) -> str:
    return json.dumps(event, ensure_ascii=False, default=str)


def _run_dict(run: Any) -> dict:
    return {
        "run_id": str(getattr(run, "id", "")),
        "objective": str(getattr(run, "objective", "")),
        "status": str(getattr(run, "status", "")),
        "error": getattr(run, "error", None),
        "created_at": getattr(run, "created_at", None),
        "updated_at": getattr(run, "updated_at", None),
    }


def _turnlog_event(entry: Any) -> Optional[dict]:
    """把 TurnLog 映射为 SSE 事件；assistant 内容已流式渲染，跳过。"""
    role = str(getattr(entry, "role", ""))
    if role == "assistant":
        return None
    if role == "tool":
        code = str(getattr(entry, "code", "") or "")
        if code == "confirmation_required":
            return None  # 由 pending 事件呈现确认卡片
        return {
            "type": "tool",
            "name": str(getattr(entry, "tool_name", "") or ""),
            "args": dict(getattr(entry, "tool_args", None) or {}),
            "result": str(getattr(entry, "tool_result", "") or "")[
                :_MAX_TOOL_RESULT_CHARS
            ],
            "code": code,
        }
    if role in ("verification", "error"):
        return {
            "type": role,
            "message": str(getattr(entry, "content", "") or ""),
            "code": str(getattr(entry, "code", "") or ""),
        }
    return None


def _emit_log(entry: Any, emit: Callable[[Optional[dict]], None]) -> None:
    event = _turnlog_event(entry)
    if event is not None:
        emit(event)


def _finalize_turn(
    session: WebAgentSession, logs: list, emit: Callable[[Optional[dict]], None]
) -> None:
    """回合收尾：run 关联、待确认票据与终态事件。"""
    session.last_active = time.time()
    run_id = next(
        (
            str(getattr(entry, "run_id", ""))
            for entry in reversed(logs)
            if getattr(entry, "run_id", None)
        ),
        None,
    )
    if run_id:
        session.run_id = run_id
        emit({"type": "run", "run_id": run_id})
    pending = getattr(session.ctx, "pending_action", None)
    if pending is not None:
        emit(
            {
                "type": "pending",
                "name": str(getattr(pending, "name", "")),
                "summary": pending_action_description(
                    session.ctx, include_local_paths=False
                ),
                "digest": str(getattr(pending, "digest", "")),
                "action_id": str(getattr(pending, "action_id", "")),
            }
        )
        emit({"type": "complete", "status": "awaiting_confirmation"})
        emit(None)
        return
    status = "done"
    error = None
    if run_id:
        record = session.ctx.store.get_agent_run(run_id)
        if record is not None:
            status = str(getattr(record, "status", "done"))
            error = getattr(record, "error", None)
    emit({"type": "complete", "status": status, "error": error})
    emit(None)


def _run_turn(
    session: WebAgentSession,
    emit: Callable[[Optional[dict]], None],
    *,
    objective: Optional[str] = None,
) -> None:
    """在独立线程执行一轮受控对话，事件经 emit 压入 SSE 队列。"""
    create_run = objective is not None
    resume_run_id = None if create_run else session.run_id

    def worker() -> None:
        try:
            new_messages, logs = chat_turn(
                session.ctx.llm,
                session.messages,
                session.ctx,
                objective=objective,
                create_run=create_run,
                run_id=resume_run_id,
                on_delta=lambda piece: emit(
                    {"type": "assistant_delta", "text": piece}
                ),
                on_log=lambda entry: _emit_log(entry, emit),
            )
        except Exception as exc:
            emit({"type": "error", "message": f"Agent 调用失败：{exc}"})
            emit(None)
            return
        session.messages = new_messages
        _finalize_turn(session, logs, emit)

    threading.Thread(target=worker, daemon=True).start()


def _sse_endpoint(
    session: WebAgentSession,
    aq: asyncio.Queue,
    emit: Callable[[Optional[dict]], None],
    *,
    start_worker: Callable[[Callable[[Optional[dict]], None]], None],
):
    """构造一个 SSE 流：start_worker 启动后台线程，队列哨兵结束并释放会话锁。

    ``emit`` 经 ``loop.call_soon_threadsafe`` 压入队列，可安全地从工作线程
    调用；流在收到 ``None`` 哨兵（或客户端断开）时结束并释放会话锁。
    """

    async def stream():
        try:
            while True:
                event = await aq.get()
                if event is None:
                    break
                yield f"data: {_event_json(event)}\n\n"
        finally:
            session.lock.release()

    try:
        start_worker(emit)
    except Exception:
        session.lock.release()
        raise
    return StreamingResponse(stream(), media_type="text/event-stream", headers=_SSE_HEADERS)


def _queue_and_emit(loop: asyncio.AbstractEventLoop) -> tuple[asyncio.Queue, Callable]:
    aq: asyncio.Queue = asyncio.Queue()

    def emit(event):
        try:
            loop.call_soon_threadsafe(aq.put_nowait, event)
        except RuntimeError:
            pass  # 事件循环已关闭（客户端断开/服务停止），丢弃迟到事件

    return aq, emit


def _session_from_payload(registry: _SessionRegistry, payload: dict) -> WebAgentSession:
    session_id = str((payload or {}).get("session_id") or "").strip()
    if len(session_id) > _MAX_SESSION_ID_CHARS:
        raise HTTPException(status_code=400, detail="session_id 过长")
    session = registry.get(session_id) if session_id else registry.get("default")
    return session


def register_agent_api(app, *, store_factory, embedder_factory, llm_factory) -> None:
    registry = _SessionRegistry(store_factory, embedder_factory, llm_factory)

    @app.post("/api/agent/chat")
    async def api_agent_chat(payload: dict):
        question = str((payload or {}).get("question") or "").strip()
        if not question:
            raise HTTPException(status_code=400, detail="question 不能为空")
        if len(question) > _MAX_QUESTION_CHARS:
            raise HTTPException(status_code=413, detail=f"question 超过 {_MAX_QUESTION_CHARS} 字符限制")
        session = _session_from_payload(registry, payload or {})
        if getattr(session.ctx, "pending_action", None) is not None:
            raise HTTPException(
                status_code=409, detail="有待确认的操作，请先在界面上确认或取消"
            )
        if not session.lock.acquire(blocking=False):
            raise HTTPException(status_code=409, detail="该会话正在处理中，请稍候")
        loop = asyncio.get_running_loop()
        aq, emit = _queue_and_emit(loop)
        emit({"type": "session", "session_id": session.id})

        def start_worker(emit):
            _run_turn(session, emit, objective=question)

        return _sse_endpoint(session, aq, emit, start_worker=start_worker)

    @app.post("/api/agent/confirm")
    async def api_agent_confirm(payload: dict):
        confirm = bool((payload or {}).get("confirm", True))
        session_id = str((payload or {}).get("session_id") or "").strip()
        session = registry.find(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="会话不存在或已过期")
        pending = getattr(session.ctx, "pending_action", None)
        if pending is None:
            raise HTTPException(status_code=400, detail="没有待确认的操作")
        if not session.lock.acquire(blocking=False):
            raise HTTPException(status_code=409, detail="该会话正在处理中，请稍候")
        loop = asyncio.get_running_loop()
        aq, emit = _queue_and_emit(loop)
        emit({"type": "session", "session_id": session.id})

        def start_worker(emit):
            def worker():
                try:
                    if confirm:
                        name, result = confirm_pending_action(session.ctx)
                        emit(
                            {
                                "type": "tool",
                                "name": str(name),
                                "args": {},
                                "result": str(result)[:_MAX_TOOL_RESULT_CHARS],
                                "code": "confirmed",
                            }
                        )
                        confirmed = getattr(session.ctx, "last_confirmed_action", None)
                        resume_run_id = (
                            getattr(confirmed, "run_id", None)
                            or getattr(pending, "run_id", None)
                            or session.run_id
                        )
                        session.run_id = str(resume_run_id) if resume_run_id else session.run_id
                        new_messages, logs = chat_turn(
                            session.ctx.llm,
                            session.messages,
                            session.ctx,
                            run_id=session.run_id,
                            on_delta=lambda piece: emit(
                                {"type": "assistant_delta", "text": piece}
                            ),
                            on_log=lambda entry: _emit_log(entry, emit),
                        )
                    else:
                        new_messages, logs = cancel_pending_run(
                            session.messages,
                            session.ctx,
                            pending=pending,
                            reason="用户在 Web 界面取消了待确认操作",
                        )
                    session.messages = new_messages
                except Exception as exc:
                    emit({"type": "error", "message": f"确认操作失败：{exc}"})
                    emit(None)
                    return
                _finalize_turn(session, logs, emit)

            threading.Thread(target=worker, daemon=True).start()

        return _sse_endpoint(session, aq, emit, start_worker=start_worker)

    @app.get("/api/agent/runs")
    def api_agent_runs(
        limit: int = Query(50, ge=1, le=200),
        offset: int = Query(0, ge=0),
    ):
        runs = registry.store.list_agent_runs(limit, offset)
        return {"items": [_run_dict(run) for run in runs]}

    @app.get("/api/agent/runs/{run_id}/events")
    def api_agent_run_events(
        run_id: str,
        after_seq: int = Query(0, ge=0),
        limit: int = Query(500, ge=1, le=1000),
    ):
        store = registry.store
        record = store.get_agent_run(run_id)
        if record is None:
            raise HTTPException(status_code=404, detail="run 不存在")
        events = store.list_agent_events(run_id, after_seq=after_seq, limit=limit)
        return {
            "run": _run_dict(record),
            "items": [
                {
                    "seq": event.seq,
                    "event_type": event.event_type,
                    "payload": event.payload,
                    "created_at": event.created_at,
                }
                for event in events
            ],
        }
