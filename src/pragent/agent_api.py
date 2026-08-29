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

会话、project 绑定、完整 transcript 与冻结的待确认票据由 Store 原子持久化；
进程内仅保留最多 64 个热会话。服务重启后仍可安全地确认或取消同一动作。

并发与断开合同：每回合持有会话排他锁，SSE 流提前断开（客户端关闭页面）
只置位 ``_TurnScope.cancel_event``——worker 在阶段边界终止并把 run 转入
``cancelled``、消息恢复到上一个持久化边界——排他锁等 worker 线程真正
结束后才释放，因此断开不会让并发回合同时修改同一 session。断开后迟到
的 SSE 事件在 emit 入口丢弃；终态以 Store 中的 run/session 记录为准。
"""
import asyncio
import json
import threading
import time
from collections import OrderedDict
from typing import Any, Callable, Optional

from fastapi import HTTPException, Query
from fastapi.responses import StreamingResponse

from .chat import TurnCancelled, cancel_pending_run, chat_turn
from .tool_protocol import PendingAction
from .tools import ToolContext, confirm_pending_action, pending_action_description

_SESSION_CAP = 64
_MAX_QUESTION_CHARS = 20_000
_MAX_SESSION_ID_CHARS = 128
_MAX_TOOL_RESULT_CHARS = 500

_SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


class _TurnScope:
    """一回合的取消与排他权协调。

    SSE 流断开时 worker 可能仍在执行：排他锁必须等 worker 真正结束后才
    释放，否则并发回合会同时修改同一 session（disconnect session race）。
    断开时置位取消事件，worker 在阶段边界终止；迟到事件在 emit 入口
    丢弃，run/session 终态由 worker 持久化到 Store。
    """

    def __init__(self, session: "WebAgentSession"):
        self.session = session
        self.cancel_event = threading.Event()
        self._worker_done = threading.Event()
        self._mutex = threading.Lock()
        self._released = False
        self._emit: Optional[Callable[[Optional[dict]], None]] = None

    def bind_emitter(self, emit: Callable[[Optional[dict]], None]) -> None:
        self._emit = emit

    def request_cancel(self) -> None:
        self.cancel_event.set()

    def stream_exited(self) -> None:
        """SSE 流结束（正常收尾或客户端断开）时调用。"""
        self.request_cancel()
        self._release()

    def worker_finished(self) -> None:
        """worker 线程真正结束：先标记完成并释放锁，再发送流终止哨兵。

        先标记后发哨兵，保证正常路径下流退出时锁立即可以释放，客户端
        收到完整响应后马上开始下一回合不会误撞 409。
        """
        self._worker_done.set()
        self._release()
        if self._emit is not None:
            self._emit(None)

    def _release(self) -> None:
        with self._mutex:
            if self._worker_done.is_set() and not self._released:
                self._released = True
                try:
                    self.session.lock.release()
                except RuntimeError:
                    # 排他锁由端点在启动 worker 前获取；直接驱动 worker 的
                    # 测试路径可能未获取锁，此时无需释放。
                    pass


def _finish_turn_scope(
    scope: Optional[_TurnScope], emit: Callable[[Optional[dict]], None]
) -> None:
    """worker 收尾：优先走 scope（标记结束 + 释放锁 + 哨兵）。"""
    if scope is not None:
        scope.worker_finished()
    else:
        emit(None)


class WebAgentSession:
    def __init__(
        self,
        session_id: str,
        store,
        embedder,
        llm,
        *,
        messages=None,
        project_id=None,
        research_repository=None,
        pending_action=None,
    ):
        self.id = session_id
        self.messages: list[dict] = list(messages or [])
        self.ctx = ToolContext(
            store=store,
            embedder=embedder,
            llm=llm,
            session_id=session_id,
            project_id=project_id,
            research_repository=research_repository,
            pending_action=pending_action,
        )
        self.lock = threading.Lock()
        self.last_active = time.time()
        self.run_id: Optional[str] = None


class _SessionRegistry:
    """会话注册表；store/embedder/llm 单例惰性创建。"""

    def __init__(self, store_factory, embedder_factory, llm_factory, repository_factory=None):
        self._store_factory = store_factory
        self._embedder_factory = embedder_factory
        self._llm_factory = llm_factory
        self._repository_factory = repository_factory
        self._store = None
        self._embedder = None
        self._llm = None
        self._repository = None
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

    @property
    def repository(self):
        if self._repository is None and self._repository_factory is not None:
            self._repository = self._repository_factory()
        return self._repository

    @staticmethod
    def _restore_pending(raw: Optional[dict]) -> Optional[PendingAction]:
        if raw is None:
            return None
        pending = PendingAction(
            name=str(raw.get("name") or ""),
            args=dict(raw.get("args") or {}),
            action_id=str(raw.get("action_id") or ""),
            digest=str(raw.get("digest") or ""),
            tool_call_id=str(raw.get("tool_call_id") or "") or None,
            run_id=str(raw.get("run_id") or "") or None,
        )
        return pending if pending.is_bound() else None

    def get(self, session_id: str, *, project_id: Optional[str] = None) -> WebAgentSession:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                state = self.store.load_agent_session_state(session_id)
                if state is None:
                    state = self.store.ensure_agent_session(
                        session_id, project_id=project_id
                    )
                    state["messages"] = []
                    state["pending_action"] = None
                elif project_id is not None and state.get("project_id") != project_id:
                    raise ValueError("Agent session 已绑定其他 project")
                embedder, llm = self._dependencies()
                pending = self._restore_pending(state.get("pending_action"))
                if state.get("pending_action") is not None and pending is None:
                    self.store.resolve_pending_action(
                        str(state["pending_action"].get("action_id") or ""), "expired",
                        error="持久化确认票据摘要不匹配",
                    )
                session = WebAgentSession(
                    session_id,
                    self.store,
                    embedder,
                    llm,
                    messages=state.get("messages") or [],
                    project_id=state.get("project_id"),
                    research_repository=self.repository,
                    pending_action=pending,
                )
                if pending is not None:
                    session.run_id = pending.run_id
                self._sessions[session_id] = session
                while len(self._sessions) > _SESSION_CAP:
                    self._sessions.popitem(last=False)
            else:
                if project_id is not None and session.ctx.project_id != project_id:
                    raise ValueError("Agent session 已绑定其他 project")
                self._sessions.move_to_end(session_id)
            session.last_active = time.time()
            return session

    def find(self, session_id: str) -> Optional[WebAgentSession]:
        with self._lock:
            return self._sessions.get(session_id)

    def find_or_restore(self, session_id: str) -> Optional[WebAgentSession]:
        session = self.find(session_id)
        if session is not None:
            return session
        if self.store.load_agent_session_state(session_id) is None:
            return None
        return self.get(session_id)

    def discard(
        self,
        session_id: str,
        *,
        expected: Optional[WebAgentSession] = None,
    ) -> None:
        """移除指定热会话；expected 防止误删并发创建的新实例。"""

        with self._lock:
            current = self._sessions.get(session_id)
            if current is not None and (expected is None or current is expected):
                self._sessions.pop(session_id, None)


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
        "project_id": getattr(run, "project_id", None),
        "session_id": getattr(run, "session_id", None),
    }


def _history_events(messages: list[dict]) -> list[dict]:
    """把模型 transcript 投影为可安全重绘的有限 UI 卡片。"""

    calls: dict[str, tuple[str, dict]] = {}
    events: list[dict] = []
    for message in messages:
        role = str(message.get("role") or "")
        if role == "user":
            events.append({"type": "message", "role": "user", "content": str(message.get("content") or "")})
        elif role == "assistant":
            content = str(message.get("content") or "")
            if content:
                events.append({"type": "message", "role": "assistant", "content": content})
            for call in message.get("tool_calls") or []:
                function = call.get("function") or {}
                try:
                    args = json.loads(function.get("arguments") or "{}")
                except (TypeError, json.JSONDecodeError):
                    args = {}
                calls[str(call.get("id") or "")] = (str(function.get("name") or ""), args)
        elif role == "tool":
            name, args = calls.get(str(message.get("tool_call_id") or ""), ("tool", {}))
            events.append(
                {
                    "type": "tool",
                    "name": name,
                    "args": args,
                    "result": str(message.get("content") or "")[:_MAX_TOOL_RESULT_CHARS],
                    "code": "restored",
                }
            )
    return events


def _pending_event(session: WebAgentSession) -> Optional[dict]:
    pending = getattr(session.ctx, "pending_action", None)
    if pending is None:
        return None
    return {
        "type": "pending",
        "name": pending.name,
        "summary": pending_action_description(session.ctx, include_local_paths=False),
        "digest": pending.digest,
        "action_id": pending.action_id,
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
        return
    status = "done"
    error = None
    if run_id:
        record = session.ctx.store.get_agent_run(run_id)
        if record is not None:
            status = str(getattr(record, "status", "done"))
            error = getattr(record, "error", None)
    emit({"type": "complete", "status": status, "error": error})


def _persist_session_state(
    session: WebAgentSession,
    emit: Callable[[Optional[dict]], None],
) -> bool:
    """原子保存 transcript 与待确认票据，并把故障显式呈现。"""

    pending = getattr(session.ctx, "pending_action", None)
    pending_payload = None
    if pending is not None:
        pending_payload = {
            "name": pending.name,
            "args": pending.args,
            "action_id": pending.action_id,
            "digest": pending.digest,
            "tool_call_id": pending.tool_call_id,
            "run_id": pending.run_id,
        }
    try:
        session.ctx.store.save_agent_session_state(
            session.id,
            session.messages,
            project_id=session.ctx.project_id,
            pending_action=pending_payload,
        )
    except Exception:
        emit(
            {
                "type": "error",
                "message": "会话状态保存失败；为避免重复操作，请勿确认并检查服务日志。",
                "code": "session_state_save_failed",
            }
        )
        return False
    return True


def _run_turn(
    session: WebAgentSession,
    emit: Callable[[Optional[dict]], None],
    *,
    objective: Optional[str] = None,
    scope: Optional[_TurnScope] = None,
) -> None:
    """在独立线程执行一轮受控对话，事件经 emit 压入 SSE 队列。

    排他锁在 worker 线程真正结束（``scope.worker_finished``）时释放；
    SSE 流提前断开只置位取消事件，不释放锁。客户端断开时回合在阶段
    边界终止：消息恢复到上一个持久化边界，run 终态已由 chat_turn 写入
    Store，迟到的 SSE 事件被丢弃。
    """
    create_run = objective is not None
    resume_run_id = None if create_run else session.run_id
    previous_messages = list(session.messages)

    def worker() -> None:
        try:
            if scope is not None:
                # 绑定本轮全新取消事件；上一轮迟到的取消信号不会污染本轮。
                session.ctx.cancel_event = scope.cancel_event
            if objective is not None:
                # objective 用于 run 审计；用户问题仍必须进入模型消息上下文。
                session.messages.append({"role": "user", "content": objective})
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
            except TurnCancelled:
                # 普通回合未产生副作用工具执行，恢复到上一个已持久化边界，
                # 避免下次请求看到半截 user turn。
                session.messages = previous_messages
                _persist_session_state(session, emit)
                emit(
                    {
                        "type": "complete",
                        "status": "cancelled",
                        "error": "客户端断开连接，回合在阶段边界终止",
                    }
                )
                return
            except Exception as exc:
                # 普通新回合失败时恢复到上一个已持久化边界。确认续跑由调
                # 用方负责保留已执行的 tool result。
                session.messages = previous_messages
                emit({"type": "error", "message": f"Agent 调用失败：{exc}"})
                return
            session.messages = new_messages
            _persist_session_state(session, emit)
            _finalize_turn(session, logs, emit)
        except Exception as exc:
            # 收尾自身故障也必须终止 SSE 流并最终释放排他锁。
            emit(
                {
                    "type": "error",
                    "message": f"Agent 回合异常终止：{exc}",
                    "code": "turn_crashed",
                }
            )
        finally:
            # 标记 worker 结束并释放锁，最后发送流终止哨兵。
            _finish_turn_scope(scope, emit)

    threading.Thread(target=worker, daemon=True).start()


def _sse_endpoint(
    session: WebAgentSession,
    *,
    start_worker: Callable[
        [Callable[[Optional[dict]], None], _TurnScope], None
    ],
):
    """构造一个 SSE 流并接管回合排他权的最终释放。

    ``start_worker(emit, scope)`` 启动后台线程。流在收到 ``None`` 哨兵或
    客户端断开时结束：``scope.stream_exited`` 置位取消事件并尝试释放锁，
    但排他锁保证等到 ``scope.worker_finished``（worker 真正结束）后才真正
    释放。断开后迟到的 emit 事件在入口丢弃；终态由 worker 持久化到
    run/session 记录，客户端重连后可从会话状态恢复。
    """
    scope = _TurnScope(session)
    loop = asyncio.get_running_loop()
    aq: asyncio.Queue = asyncio.Queue()
    emit = _sse_emitter(loop, aq, scope)
    scope.bind_emitter(emit)

    async def stream():
        try:
            while True:
                event = await aq.get()
                if event is None:
                    break
                yield f"data: {_event_json(event)}\n\n"
        finally:
            scope.stream_exited()

    try:
        start_worker(emit, scope)
    except Exception:
        # worker 未启动成功也必须释放锁，否则会话永久 409。
        _finish_turn_scope(scope, emit)
        raise
    return StreamingResponse(stream(), media_type="text/event-stream", headers=_SSE_HEADERS)


def _sse_emitter(
    loop: asyncio.AbstractEventLoop,
    aq: asyncio.Queue,
    scope: _TurnScope,
) -> Callable[[Optional[dict]], None]:
    def emit(event):
        if scope.cancel_event.is_set():
            return  # 流已结束/客户端断开：迟到事件直接丢弃
        try:
            loop.call_soon_threadsafe(aq.put_nowait, event)
        except RuntimeError:
            pass  # 事件循环已关闭（服务停止），丢弃迟到事件

    return emit


def _session_from_payload(registry: _SessionRegistry, payload: dict) -> WebAgentSession:
    session_id = str((payload or {}).get("session_id") or "").strip()
    if len(session_id) > _MAX_SESSION_ID_CHARS:
        raise HTTPException(status_code=400, detail="session_id 过长")
    project_id = str((payload or {}).get("project_id") or "").strip() or None
    if project_id is not None and len(project_id) > 128:
        raise HTTPException(status_code=400, detail="project_id 过长")
    try:
        session = registry.get(
            session_id or "default", project_id=project_id
        )
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return session


def _run_confirmation(
    session: WebAgentSession,
    emit: Callable[[Optional[dict]], None],
    *,
    pending: PendingAction,
    confirm: bool,
    scope: Optional[_TurnScope] = None,
) -> None:
    """在独立线程执行确认/取消续跑；收尾合同与 ``_run_turn`` 一致。"""

    def worker() -> None:
        try:
            if scope is not None:
                session.ctx.cancel_event = scope.cancel_event
            try:
                if confirm:
                    if not session.ctx.store.claim_pending_action(pending.action_id):
                        emit(
                            {
                                "type": "error",
                                "message": "待确认操作已被其他请求处理，请刷新会话状态。",
                                "code": "confirmation_already_claimed",
                            }
                        )
                        return
                    name, result = confirm_pending_action(session.ctx)
                    if not session.ctx.store.resolve_pending_action(
                        pending.action_id, "executed", result=result
                    ):
                        raise RuntimeError("待确认操作已被其他请求处理")
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
                    if not session.ctx.store.resolve_pending_action(
                        pending.action_id, "rejected",
                        error="用户在 Web 界面取消了待确认操作",
                    ):
                        raise RuntimeError("待确认操作已被其他请求处理")
                session.messages = new_messages
            except TurnCancelled:
                # 确认可能已执行有副作用的工具：协议已闭合的 transcript 必须
                # 保留，绝不回滚成"从未发生"。
                _persist_session_state(session, emit)
                emit(
                    {
                        "type": "complete",
                        "status": "cancelled",
                        "error": "客户端断开连接，回合在阶段边界终止",
                    }
                )
                return
            except Exception as exc:
                # 确认可能已经执行了有副作用的工具。若协议已经闭合，必须
                # 保存 tool result，避免重启后把已执行动作伪装成从未发生。
                _persist_session_state(session, emit)
                emit({"type": "error", "message": f"确认操作失败：{exc}"})
                return
            _persist_session_state(session, emit)
            _finalize_turn(session, logs, emit)
        except Exception as exc:
            emit(
                {
                    "type": "error",
                    "message": f"Agent 回合异常终止：{exc}",
                    "code": "turn_crashed",
                }
            )
        finally:
            # 标记 worker 结束并释放锁，最后发送流终止哨兵。
            _finish_turn_scope(scope, emit)

    threading.Thread(target=worker, daemon=True).start()


def register_agent_api(
    app, *, store_factory, embedder_factory, llm_factory, repository_factory=None
) -> None:
    registry = _SessionRegistry(
        store_factory, embedder_factory, llm_factory, repository_factory
    )

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

        def start_worker(emit, scope):
            emit({"type": "session", "session_id": session.id})
            _run_turn(session, emit, objective=question, scope=scope)

        return _sse_endpoint(session, start_worker=start_worker)

    @app.post("/api/agent/confirm")
    async def api_agent_confirm(payload: dict):
        confirm = bool((payload or {}).get("confirm", True))
        session_id = str((payload or {}).get("session_id") or "").strip()
        session = registry.find_or_restore(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="会话不存在或已过期")
        pending = getattr(session.ctx, "pending_action", None)
        if pending is None:
            raise HTTPException(status_code=400, detail="没有待确认的操作")
        if not session.lock.acquire(blocking=False):
            raise HTTPException(status_code=409, detail="该会话正在处理中，请稍候")

        def start_worker(emit, scope):
            emit({"type": "session", "session_id": session.id})
            _run_confirmation(session, emit, pending=pending, confirm=confirm, scope=scope)

        return _sse_endpoint(session, start_worker=start_worker)

    @app.get("/api/agent/sessions/{session_id}")
    def api_agent_session(session_id: str):
        normalized_id = str(session_id or "").strip()
        if not normalized_id or len(normalized_id) > _MAX_SESSION_ID_CHARS:
            raise HTTPException(status_code=400, detail="session_id 不合法")
        session = registry.find_or_restore(normalized_id)
        if session is None:
            raise HTTPException(status_code=404, detail="会话不存在")
        return {
            "session_id": session.id,
            "project_id": session.ctx.project_id,
            "history": _history_events(session.messages),
            "pending": _pending_event(session),
            "run_id": session.run_id,
        }

    @app.delete("/api/agent/sessions/{session_id}")
    def api_agent_clear_session(session_id: str):
        normalized_id = str(session_id or "").strip()
        if not normalized_id:
            raise HTTPException(status_code=400, detail="session_id 不能为空")
        if len(normalized_id) > _MAX_SESSION_ID_CHARS:
            raise HTTPException(status_code=400, detail="session_id 过长")

        session = registry.find_or_restore(normalized_id)
        if session is None:
            return {
                "session_id": normalized_id,
                "cleared": registry.store.delete_agent_session(normalized_id),
            }
        if not session.lock.acquire(blocking=False):
            raise HTTPException(status_code=409, detail="该会话正在处理中，不能清空")
        try:
            if getattr(session.ctx, "pending_action", None) is not None:
                raise HTTPException(status_code=409, detail="会话有待确认操作，不能清空")
            cleared = registry.store.delete_agent_session(normalized_id)
            session.messages = []
            session.run_id = None
            registry.discard(normalized_id, expected=session)
        finally:
            session.lock.release()
        return {"session_id": normalized_id, "cleared": cleared}

    @app.get("/api/agent/runs")
    def api_agent_runs(
        limit: int = Query(50, ge=1, le=200),
        offset: int = Query(0, ge=0),
        project_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ):
        runs = registry.store.list_agent_runs(
            limit, offset, project_id=project_id, session_id=session_id
        )
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
