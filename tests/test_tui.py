import asyncio

from paper_agent.models import Chunk
from paper_agent.store import Store
from paper_agent.tools import ToolContext
from paper_agent.tui import ChatApp
from textual.widgets import Input, RichLog

from helpers import FakeEmbedder, make_paper


class FakeLLM:
    is_configured = True

    def __init__(self, script):
        self.script = list(script)

    def chat_with_tools(self, system, messages, tools):
        return self.script.pop(0)


def make_ctx(tmp_path):
    s = Store(tmp_path / "t.db")
    pid = s.upsert_paper(make_paper("a.pdf", title="注意力机制研究", year=2023))
    s.replace_chunks(
        pid, [Chunk(None, pid, 0, 1, "注意力机制是文本分类的关键技术。", FakeEmbedder.vecs_for("x"))]
    )
    return ToolContext(store=s, embedder=FakeEmbedder(), llm=FakeLLM([]))


def log_text(app) -> str:
    log = app.query_one("#log", RichLog)
    return "\n".join(strip.text for strip in log.lines)


def test_chat_app_help_command(tmp_path):
    async def run():
        ctx = make_ctx(tmp_path)
        ctx.pending_action = ("save_note", {"filename": "stale.md", "content": "stale"})
        app = ChatApp(llm=ctx.llm, ctx=ctx)
        async with app.run_test() as pilot:
            inp = app.query_one(Input)
            inp.focus()
            await pilot.pause(0.1)
            inp.value = "/help"
            await pilot.press("enter")
            await pilot.pause(0.2)
            return log_text(app)

    text = asyncio.run(run())
    assert "local_search" in text
    assert "download_paper" in text


def test_chat_app_tool_flow(tmp_path):
    llm = FakeLLM(
        [
            {"content": None, "tool_calls": [{"id": "c1", "name": "library_status", "arguments": {}}]},
            {"content": "库里有 1 篇论文。", "tool_calls": []},
        ]
    )

    async def run():
        ctx = make_ctx(tmp_path)
        ctx.llm = llm
        app = ChatApp(llm=llm, ctx=ctx)
        async with app.run_test() as pilot:
            inp = app.query_one(Input)
            inp.focus()
            await pilot.pause(0.1)
            inp.value = "库里有什么？"
            await pilot.press("enter")
            # 等待响应完成（Input 恢复可用）
            for _ in range(100):
                await pilot.pause(0.1)
                if not app.query_one(Input).disabled:
                    break
            return log_text(app)

    text = asyncio.run(run())
    assert "你：库里有什么？" in text
    assert "工具 library_status" in text
    assert "论文 1 篇" in text
    assert "库里有 1 篇论文。" in text


def test_chat_app_clear(tmp_path):
    async def run():
        ctx = make_ctx(tmp_path)
        app = ChatApp(llm=ctx.llm, ctx=ctx)
        async with app.run_test() as pilot:
            inp = app.query_one(Input)
            inp.focus()
            await pilot.pause(0.1)
            inp.value = "/help"
            await pilot.press("enter")
            await pilot.pause(0.2)
            first = log_text(app)
            inp.value = "/clear"
            await pilot.press("enter")
            await pilot.pause(0.2)
            return first, log_text(app), ctx.pending_action

    first, cleared, pending = asyncio.run(run())
    assert first != ""
    assert cleared == ""
    assert pending is None


def test_chat_app_confirm_executes_pending_exact_action(tmp_path, monkeypatch):
    notes = tmp_path / "notes"
    monkeypatch.setattr("paper_agent.config.notes_dir", lambda: notes)

    async def run():
        ctx = make_ctx(tmp_path)
        ctx.pending_action = ("save_note", {"filename": "confirmed.md", "content": "ok"})
        app = ChatApp(llm=ctx.llm, ctx=ctx)
        async with app.run_test() as pilot:
            inp = app.query_one(Input)
            inp.focus()
            await pilot.pause(0.1)
            inp.value = "/confirm"
            await pilot.press("enter")
            for _ in range(100):
                await pilot.pause(0.05)
                if not app.query_one(Input).disabled:
                    break
            return log_text(app), ctx.pending_action

    text, pending = asyncio.run(run())
    assert pending is None
    assert (notes / "confirmed.md").read_text(encoding="utf-8") == "ok"
    assert "已确认工具 save_note" in text


def test_chat_app_blocks_new_question_while_action_is_pending(tmp_path):
    async def run():
        ctx = make_ctx(tmp_path)
        ctx.pending_action = ("save_note", {"filename": "pending.md", "content": "x"})
        app = ChatApp(llm=ctx.llm, ctx=ctx)
        async with app.run_test() as pilot:
            inp = app.query_one(Input)
            inp.focus()
            await pilot.pause(0.1)
            inp.value = "换一个操作"
            await pilot.press("enter")
            await pilot.pause(0.2)
            return log_text(app), app._messages, ctx.pending_action

    text, messages, pending = asyncio.run(run())
    assert "请先输入 /confirm 或 /cancel" in text
    assert messages == []
    assert pending is not None


def test_chat_app_copy_last_answer(tmp_path, monkeypatch):
    import pyperclip

    copied = []
    monkeypatch.setattr(pyperclip, "copy", lambda t: copied.append(t))
    llm = FakeLLM(
        [
            {"content": None, "tool_calls": [{"id": "c1", "name": "library_status", "arguments": {}}]},
            {"content": "库里有 1 篇论文。", "tool_calls": []},
        ]
    )

    async def run():
        ctx = make_ctx(tmp_path)
        ctx.llm = llm
        app = ChatApp(llm=llm, ctx=ctx)
        async with app.run_test() as pilot:
            inp = app.query_one(Input)
            inp.focus()
            await pilot.pause(0.1)
            inp.value = "库里有什么？"
            await pilot.press("enter")
            for _ in range(100):
                await pilot.pause(0.1)
                if not app.query_one(Input).disabled:
                    break
            inp.value = "/copy"
            await pilot.press("enter")
            await pilot.pause(0.2)
            return log_text(app)

    text = asyncio.run(run())
    assert copied == ["库里有 1 篇论文。"]
    assert "已复制到剪贴板" in text


def test_chat_app_copy_no_answer(tmp_path, monkeypatch):
    import pyperclip

    copied = []
    monkeypatch.setattr(pyperclip, "copy", lambda t: copied.append(t))

    async def run():
        ctx = make_ctx(tmp_path)
        app = ChatApp(llm=ctx.llm, ctx=ctx)
        async with app.run_test() as pilot:
            inp = app.query_one(Input)
            inp.focus()
            await pilot.pause(0.1)
            inp.value = "/copy"
            await pilot.press("enter")
            await pilot.pause(0.2)
            return log_text(app)

    text = asyncio.run(run())
    assert copied == []
    assert "没有可复制的回答" in text


def test_chat_app_export(tmp_path, monkeypatch):
    notes = tmp_path / "notes"
    monkeypatch.setattr("paper_agent.config.notes_dir", lambda: notes)
    llm = FakeLLM(
        [
            {"content": None, "tool_calls": [{"id": "c1", "name": "library_status", "arguments": {}}]},
            {"content": "库里有 1 篇论文。", "tool_calls": []},
        ]
    )

    async def run():
        ctx = make_ctx(tmp_path)
        ctx.llm = llm
        app = ChatApp(llm=llm, ctx=ctx)
        async with app.run_test() as pilot:
            inp = app.query_one(Input)
            inp.focus()
            await pilot.pause(0.1)
            inp.value = "库里有什么？"
            await pilot.press("enter")
            for _ in range(100):
                await pilot.pause(0.1)
                if not app.query_one(Input).disabled:
                    break
            inp.value = "/export"
            await pilot.press("enter")
            await pilot.pause(0.2)
            return log_text(app)

    text = asyncio.run(run())
    assert "对话已导出" in text
    exported = list(notes.glob("chat-*.md"))
    assert len(exported) == 1
    content = exported[0].read_text(encoding="utf-8")
    assert "**你**：库里有什么？" in content
    assert "**助手**：库里有 1 篇论文。" in content
    assert "library_status" in content
