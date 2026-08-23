"""textual TUI：PRAgent 论文助手对话界面（模型可调用工具，流式回答）。"""
import asyncio
import threading
from datetime import datetime

from rich.markup import escape
from rich.text import Text
from textual.app import App, ComposeResult
from textual.widgets import Footer, Header, Input, RichLog, Static

from . import __version__
from .chat import cancel_pending_run, chat_turn
from .tools import (
    ToolContext,
    cancel_pending_action,
    confirm_pending_action,
    pending_action_description,
)


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


class ChatApp(App):
    TITLE = "PRAgent"
    SUB_TITLE = f"v{__version__} 对话模式"

    CSS = """
    #log {
        height: 1fr;
        border: round $primary;
        padding: 1;
    }
    #streaming {
        display: none;
        border: round $success;
        padding: 1 2;
        margin-top: 1;
        color: $success;
    }
    #input {
        dock: bottom;
        margin-top: 1;
    }
    """

    def __init__(self, llm, ctx: ToolContext):
        super().__init__()
        self._llm = llm
        self._ctx = ctx
        self._messages: list[dict] = []
        # 流式回答：LLM 线程写入缓冲区，增量经 call_from_thread 同步渲染。
        self._stream_lock = threading.Lock()
        self._stream_buffer: list[str] = []

    def compose(self) -> ComposeResult:
        yield Header()
        yield RichLog(id="log", highlight=True, markup=True, wrap=True)
        yield Static(id="streaming", expand=True)
        yield Input(id="input", placeholder="输入问题；/help /confirm /cancel /quit")

    def on_mount(self) -> None:
        self._log(
            "[cyan]PRAgent 论文助手对话模式：模型可自动调用工具"
            "（检索 / 深读 / 固定证据 / arXiv / 下载 / 索引），回答逐字流式输出。"
            "输入 /help 查看命令。[/cyan]"
        )

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        self.query_one(Input).value = ""
        if not text:
            return
        action = self._handle_command(text)
        if action == "quit":
            self.exit()
            return
        if action == "handled":
            return
        if self._ctx.pending_action is not None:
            self._log("[yellow]当前仍有待确认操作；请先输入 /confirm 或 /cancel。[/yellow]")
            return
        self._messages.append({"role": "user", "content": text})
        self._log(f"[bold blue]你：{escape(text)}[/bold blue]\n")
        self.query_one(Input).disabled = True
        self.run_worker(
            self._respond(objective=text, create_run=True),
            group="chat",
        )

    def _handle_command(self, text: str):
        if text in ("/quit", "/exit"):
            return "quit"
        if text == "/clear":
            try:
                self._cancel_for_ui(log_result=False)
            except Exception as exc:
                self._log(f"[red]取消当前 Agent run 失败：{escape(str(exc))}[/red]")
                return "handled"
            self._messages.clear()
            if hasattr(self._ctx, "last_confirmed_action"):
                self._ctx.last_confirmed_action = None
            self.query_one("#log", RichLog).clear()
            return "handled"
        if text == "/export":
            return self._export_chat()
        if text == "/copy":
            return self._copy_last_answer()
        if text == "/confirm":
            if self._ctx.pending_action is None:
                self._log("[yellow]没有待确认的工具操作。[/yellow]")
                return "handled"
            self.query_one(Input).disabled = True
            self.run_worker(self._confirm_pending(), group="chat")
            return "handled"
        if text == "/cancel":
            try:
                if not self._cancel_for_ui(log_result=True):
                    self._log("[yellow]没有待确认的工具操作。[/yellow]")
            except Exception as exc:
                self._log(f"[red]取消当前 Agent run 失败：{escape(str(exc))}[/red]")
            return "handled"
        if text == "/help":
            self._log(
                "[yellow]命令：/clear 清空对话，/copy 复制最后一条回答，/export 导出对话，"
                "/confirm 执行待确认写操作，/cancel 取消待确认操作，/quit 退出。"
                "工具：local_search 本地检索 / web_search arXiv 搜索 / download_paper 下载并索引 / "
                "index_papers 索引目录 / list_papers 论文列表 / library_status 库状态 / "
                "search_within_paper 单篇检索 / get_paper_outline 分页概览 / "
                "read_pages 阅读页面 / read_chunk_context 阅读上下文 / "
                "pin_evidence 固定证据 / get_evidence、list_evidence 读取证据 / "
                "save_note 保存笔记 / list_notes 笔记列表[/yellow]"
            )
            return "handled"
        return None

    async def _confirm_pending(self) -> None:
        pending = self._ctx.pending_action
        try:
            name, result = await asyncio.to_thread(confirm_pending_action, self._ctx)
            if name:
                self._log(f"[dim]→ 已确认工具 {escape(name)}[/dim]")
                confirmed = getattr(self._ctx, "last_confirmed_action", None)
                tool_call_id = getattr(confirmed, "tool_call_id", None) or getattr(
                    pending, "tool_call_id", None
                )
                run_id = getattr(confirmed, "run_id", None) or getattr(
                    pending, "run_id", None
                )
                structured_result = getattr(confirmed, "result", None)
                if tool_call_id:
                    new_messages, logs = await asyncio.to_thread(
                        chat_turn,
                        self._llm,
                        self._messages,
                        self._ctx,
                        run_id=run_id,
                        confirmed_tool_result=structured_result or result,
                        confirmed_tool_call_id=tool_call_id,
                        on_delta=self._on_answer_delta,
                    )
                    self._messages = new_messages
                    streamed = self._finish_stream()
                    self._render_logs(logs, streamed=streamed)
                else:
                    # 兼容手工注入的旧式 pending tuple；真实 Agent pending 总会
                    # 带原始 tool_call_id，并走上面的协议化续跑路径。
                    self._log(f"[green]{escape(result)}[/green]")
            else:
                self._log(f"[yellow]{result}[/yellow]")
        except Exception as exc:
            self._log(f"[red]确认操作执行失败：{escape(str(exc))}[/red]")
        finally:
            self._respond_done()

    def _copy_last_answer(self) -> str:
        """把最近一条 AI 回答复制到 Windows 剪贴板（与终端选择机制无关）。"""
        content = None
        for msg in reversed(self._messages):
            if msg["role"] == "assistant" and msg.get("content"):
                content = msg["content"]
                break
        if not content:
            self._log("[yellow]没有可复制的回答。[/yellow]")
            return "handled"
        try:
            import pyperclip

            pyperclip.copy(content)
        except Exception as exc:
            self._log(f"[red]复制失败：{escape(str(exc))}[/red]")
            return "handled"
        self._log("[green]最后一条回答已复制到剪贴板（Ctrl+V 可粘贴）。[/green]")
        return "handled"

    def _export_chat(self) -> str:
        from . import config

        if not self._messages:
            self._log("[yellow]当前没有对话内容可导出。[/yellow]")
            return "handled"
        notes = config.notes_dir()
        notes.mkdir(parents=True, exist_ok=True)
        path = notes / f"chat-{_now_stamp()}.md"
        path.write_text(self._export_markdown(), encoding="utf-8")
        self._log(f"[green]对话已导出到 {escape(str(path))}[/green]")
        return "handled"

    def _export_markdown(self) -> str:
        lines = []
        for msg in self._messages:
            if msg["role"] == "user":
                lines.append(f"**你**：{msg['content']}\n")
            elif msg["role"] == "assistant":
                if msg.get("content"):
                    lines.append(f"**助手**：{msg['content']}\n")
                for tc in msg.get("tool_calls", []):
                    lines.append(f"- 调用工具 `{tc['function']['name']}({tc['function']['arguments']})`\n")
            elif msg["role"] == "tool":
                lines.append(f"  ↳ 结果：{msg['content'][:500]}\n")
        return "\n".join(lines)

    def _respond_done(self) -> None:
        """恢复输入框可用与焦点（disable/enable 会丢失焦点）。"""
        inp = self.query_one(Input)
        inp.disabled = False
        inp.focus()

    async def _respond(
        self,
        *,
        objective: str | None = None,
        create_run: bool = False,
    ) -> None:
        try:
            new_messages, logs = await asyncio.to_thread(
                chat_turn,
                self._llm,
                self._messages,
                self._ctx,
                objective=objective,
                create_run=create_run,
                on_delta=self._on_answer_delta,
            )
        except Exception as exc:
            self._finish_stream()
            self._log(f"[red]调用失败：{escape(str(exc))}[/red]")
            self._respond_done()
            return
        streamed = self._finish_stream()
        self._messages = new_messages
        self._render_logs(logs, streamed=streamed)
        if self._ctx.pending_action is not None:
            self._log(
                "[yellow]待确认："
                f"{escape(pending_action_description(self._ctx))}。"
                "输入 /confirm 执行，或 /cancel 取消。[/yellow]"
            )
        self._respond_done()

    def _on_answer_delta(self, piece: str) -> None:
        """LLM 线程回调：记录增量并同步调度到事件循环渲染。

        ``call_from_thread`` 会阻塞调用线程直到渲染完成，因此增量按到达
        顺序严格渲染，聊天线程返回前所有增量都已落盘。
        """
        with self._stream_lock:
            self._stream_buffer.append(piece)
        self.call_from_thread(self._render_stream_delta)

    def _render_stream_delta(self) -> None:
        """在 App 事件循环线程把累计增量刷新到流式面板。"""
        with self._stream_lock:
            text = "".join(self._stream_buffer)
        panel = self.query_one("#streaming", Static)
        panel.styles.display = "block"
        panel.update(Text(text, style="green"))

    def _finish_stream(self) -> str:
        """结束流式渲染并返回完整流式文本（须在 App 事件循环线程调用）。

        把面板内容一次性落入 RichLog（保持绿色助手样式），随后隐藏面板；
        流式文本会跳过日志中的 assistant 条目，避免重复展示。
        """
        with self._stream_lock:
            text = "".join(self._stream_buffer)
            self._stream_buffer = []
        panel = self.query_one("#streaming", Static)
        panel.styles.display = "none"
        panel.update("")
        if text:
            self._log(f"[green]{escape(text)}[/green]\n")
        return text

    def _render_logs(self, logs, *, streamed: str = "") -> None:
        """把 Agent 事件的用户可见部分统一渲染到 TUI。

        流式模式下模型文本（含工具轮的口头说明与最终回答）已实时渲染，
        跳过 assistant 条目避免重复。
        """
        for entry in logs:
            if entry.role == "assistant":
                if not streamed and entry.content:
                    self._log(f"[green]{escape(entry.content)}[/green]\n")
                continue
            elif entry.role == "tool":
                arg_str = ", ".join(f"{k}={v}" for k, v in (entry.tool_args or {}).items())
                if len(arg_str) > 500:
                    arg_str = arg_str[:500] + "…（完整内容见待确认摘要的长度与 SHA-256）"
                self._log(
                    f"[dim]→ 工具 {escape(entry.tool_name or '')}({escape(arg_str)})[/dim]"
                )
                head = (entry.tool_result or "").replace("\n", " ")[:200]
                self._log(f"[dim]  ↳ {escape(head)}[/dim]")
            elif entry.role == "error":
                self._log(f"[red]{escape(entry.content)}[/red]")
            elif entry.role == "verification" and entry.content:
                self._log(f"[yellow]{escape(entry.content)}[/yellow]")

    def _cancel_for_ui(self, *, log_result: bool) -> bool:
        """取消 pending；新协议优先，旧式测试/会话安全降级。"""
        pending = self._ctx.pending_action
        if pending is None:
            return False
        try:
            self._messages, logs = cancel_pending_run(
                self._messages,
                self._ctx,
                pending=pending,
            )
        except Exception:
            if getattr(pending, "tool_call_id", None) or getattr(
                pending, "run_id", None
            ):
                # 新协议的 pending 若取消失败，必须保留现场以便重试，不能把
                # 持久 run 留在 awaiting_confirmation 却清掉本地票据。
                raise
            cancel_pending_action(self._ctx)
            logs = []
        if log_result:
            if logs:
                self._render_logs(logs)
            self._log("[yellow]已取消待确认的工具操作。[/yellow]")
        return True

    def _log(self, text: str) -> None:
        self.query_one("#log", RichLog).write(text)
