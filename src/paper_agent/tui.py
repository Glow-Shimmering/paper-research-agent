"""textual TUI：论文助手对话界面（模型可调用工具）。"""
import asyncio
from datetime import datetime

from rich.markup import escape
from textual.app import App, ComposeResult
from textual.widgets import Footer, Header, Input, RichLog

from . import __version__
from .chat import chat_turn
from .tools import (
    ToolContext,
    cancel_pending_action,
    confirm_pending_action,
    pending_action_description,
)


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


class ChatApp(App):
    TITLE = "论文助手"
    SUB_TITLE = f"v{__version__} 对话模式"

    CSS = """
    #log {
        height: 1fr;
        border: round $primary;
        padding: 1;
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

    def compose(self) -> ComposeResult:
        yield Header()
        yield RichLog(id="log", highlight=True, markup=True, wrap=True)
        yield Input(id="input", placeholder="输入问题；/help /confirm /cancel /quit")

    def on_mount(self) -> None:
        self._log(
            "[cyan]论文助手对话模式：模型可自动调用工具"
            "（本地检索 / arXiv 搜索 / 下载 / 索引 / 列表 / 状态）。输入 /help 查看命令。[/cyan]"
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
        self.run_worker(self._respond(), group="chat")

    def _handle_command(self, text: str):
        if text in ("/quit", "/exit"):
            return "quit"
        if text == "/clear":
            self._messages.clear()
            cancel_pending_action(self._ctx)
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
            if cancel_pending_action(self._ctx):
                self._log("[yellow]已取消待确认的工具操作。[/yellow]")
            else:
                self._log("[yellow]没有待确认的工具操作。[/yellow]")
            return "handled"
        if text == "/help":
            self._log(
                "[yellow]命令：/clear 清空对话，/copy 复制最后一条回答，/export 导出对话，"
                "/confirm 执行待确认写操作，/cancel 取消待确认操作，/quit 退出。"
                "工具：local_search 本地检索 / web_search arXiv 搜索 / download_paper 下载并索引 / "
                "index_papers 索引目录 / list_papers 论文列表 / library_status 库状态 / "
                "save_note 保存笔记 / list_notes 笔记列表[/yellow]"
            )
            return "handled"
        return None

    async def _confirm_pending(self) -> None:
        try:
            name, result = await asyncio.to_thread(confirm_pending_action, self._ctx)
            if name:
                self._log(f"[dim]→ 已确认工具 {escape(name)}[/dim]")
                self._log(f"[green]{escape(result)}[/green]")
                self._messages.append(
                    {
                        "role": "user",
                        "content": f"[用户已确认执行工具 {name}；执行结果：{result}]",
                    }
                )
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

    async def _respond(self) -> None:
        try:
            new_messages, logs = await asyncio.to_thread(
                chat_turn, self._llm, self._messages, self._ctx
            )
        except Exception as exc:
            self._log(f"[red]调用失败：{escape(str(exc))}[/red]")
            self._respond_done()
            return
        self._messages = new_messages
        for entry in logs:
            if entry.role == "assistant" and entry.content:
                self._log(f"[green]{escape(entry.content)}[/green]\n")
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
        if self._ctx.pending_action is not None:
            self._log(
                "[yellow]待确认："
                f"{escape(pending_action_description(self._ctx))}。"
                "输入 /confirm 执行，或 /cancel 取消。[/yellow]"
            )
        self._respond_done()

    def _log(self, text: str) -> None:
        self.query_one("#log", RichLog).write(text)
