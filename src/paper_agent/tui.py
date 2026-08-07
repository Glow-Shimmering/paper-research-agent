"""textual TUI：论文助手对话界面（模型可调用工具）。"""
import asyncio

from textual.app import App, ComposeResult
from textual.widgets import Footer, Header, Input, RichLog

from . import __version__
from .chat import chat_turn
from .tools import ToolContext


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
        yield Input(id="input", placeholder="输入问题；/help /clear /quit")

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
        self._messages.append({"role": "user", "content": text})
        self._log(f"[bold blue]你：{text}[/bold blue]\n")
        self.query_one(Input).disabled = True
        self.run_worker(self._respond(), group="chat")

    def _handle_command(self, text: str):
        if text in ("/quit", "/exit"):
            return "quit"
        if text == "/clear":
            self._messages.clear()
            self.query_one("#log", RichLog).clear()
            return "handled"
        if text == "/help":
            self._log(
                "[yellow]命令：/clear 清空对话，/quit 退出。"
                "工具：local_search 本地检索 / web_search arXiv 搜索 / download_paper 下载并索引 / "
                "index_papers 索引目录 / list_papers 论文列表 / library_status 库状态[/yellow]"
            )
            return "handled"
        return None

    async def _respond(self) -> None:
        try:
            new_messages, logs = await asyncio.to_thread(
                chat_turn, self._llm, self._messages, self._ctx
            )
        except Exception as exc:
            self._log(f"[red]调用失败：{exc}[/red]")
            self.query_one(Input).disabled = False
            return
        self._messages = new_messages
        for entry in logs:
            if entry.role == "assistant" and entry.content:
                self._log(f"[green]{entry.content}[/green]\n")
            elif entry.role == "tool":
                arg_str = ", ".join(f"{k}={v}" for k, v in (entry.tool_args or {}).items())
                self._log(f"[dim]→ 工具 {entry.tool_name}({arg_str})[/dim]")
                head = (entry.tool_result or "").replace("\n", " ")[:200]
                self._log(f"[dim]  ↳ {head}[/dim]")
            elif entry.role == "error":
                self._log(f"[red]{entry.content}[/red]")
        self.query_one(Input).disabled = False

    def _log(self, text: str) -> None:
        self.query_one("#log", RichLog).write(text)
