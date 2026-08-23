"""流式输出：LLM 客户端重建、问答事件流、Agent 回调、Web SSE 与 CLI 展示。"""
from types import SimpleNamespace

from fastapi.testclient import TestClient as FastAPITestClient
from typer.testing import CliRunner

from pragent.answer import answer_stream
from pragent.chat import chat_turn
from pragent.llm import LLMClient, LLMError
from pragent.models import Chunk
from pragent.store import Store
from pragent.tools import ToolContext
from pragent.webapp import create_app

from helpers import FakeEmbedder, StreamFakeLLM, StreamingScriptLLM, make_paper


# ---------- LLMClient 流式重建 ----------


def _chunk(content=None, tool_delta=None, finish="stop", usage=None, rid="resp-1"):
    delta = SimpleNamespace(
        content=content,
        tool_calls=[tool_delta] if tool_delta is not None else None,
    )
    choice = SimpleNamespace(delta=delta, finish_reason=finish)
    return SimpleNamespace(choices=[choice], usage=usage, id=rid)


def _tool_delta(index, id=None, name=None, args=None):
    function = SimpleNamespace(name=name, arguments=args)
    return SimpleNamespace(index=index, id=id, function=function)


class _FakeCompletions:
    def __init__(self, chunks, calls):
        self.chunks = chunks
        self.calls = calls

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            return iter(self.chunks)
        return self.chunks[0]


class _FakeClient:
    def __init__(self, chunks, calls):
        self.chat = SimpleNamespace(completions=_FakeCompletions(chunks, calls))


def _stream_llm(chunks):
    calls = []
    llm = LLMClient("http://localhost", "key", "model")
    llm._client = _FakeClient(chunks, calls)
    return llm, calls


def test_chat_stream_yields_deltas_and_metadata():
    llm, calls = _stream_llm(
        [
            _chunk(content="你好"),
            _chunk(content="，"),
            _chunk(
                content="世界",
                finish="stop",
                usage={"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
            ),
        ]
    )

    pieces = list(llm.chat_stream("sys", "user"))

    assert pieces == ["你好", "，", "世界"]
    assert calls[0]["stream"] is True
    assert llm.last_response_metadata["usage"]["total_tokens"] == 3
    assert llm.last_response_metadata["finish_reason"] == "stop"


def test_chat_with_tools_stream_reassembles_tool_calls():
    llm, _ = _stream_llm(
        [
            _chunk(tool_delta=_tool_delta(0, id="call_1", name="list", args='{"limit":')),
            _chunk(
                tool_delta=_tool_delta(0, args="5}"),
                finish="tool_calls",
                usage={"prompt_tokens": 1},
            ),
        ]
    )
    seen = []

    result = llm.chat_with_tools("sys", [], [], on_delta=seen.append)

    assert result["content"] is None
    assert result["tool_calls"] == [{"id": "call_1", "name": "list", "arguments": {"limit": 5}}]
    assert seen == []


def test_chat_with_tools_plain_path_unchanged():
    message = SimpleNamespace(content="普通回答", tool_calls=[])
    resp = SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None, id="r")
    llm, calls = _stream_llm([resp])

    result = llm.chat_with_tools("sys", [], [])

    assert result["content"] == "普通回答"
    assert result["tool_calls"] == []
    assert calls[-1].get("stream") is None


def test_chat_with_tools_stream_forwards_content_deltas():
    llm, _ = _stream_llm([_chunk(content="开"), _chunk(content="头")])
    seen = []

    result = llm.chat_with_tools("sys", [], [], on_delta=seen.append)

    assert result["content"] == "开头"
    assert seen == ["开", "头"]


# ---------- answer_stream 事件流 ----------


def seed(tmp_path, texts):
    s = Store(tmp_path / "t.db")
    pid = s.upsert_paper(make_paper("a.pdf", title="论文一", year=2023))
    s.replace_chunks(
        pid,
        [Chunk(None, pid, i, 1, t, FakeEmbedder.vecs_for(t)) for i, t in enumerate(texts)],
    )
    return s, FakeEmbedder()


def test_answer_stream_emits_context_deltas_complete(tmp_path):
    s, emb = seed(tmp_path, ["注意力机制的提出背景与动机。"])
    llm = StreamFakeLLM(["回答开头 [1]，", "回答结尾 [1]。"])
    events = list(answer_stream(s, emb, llm, "这篇论文讲了什么？", top=3))

    assert events[0]["type"] == "context"
    assert events[0]["retrieval_only"] is False
    assert len(events[0]["sources"]) == 2  # 命中片段 + 库藏目录
    assert events[0]["sources"][1]["catalog"] is True
    deltas = [e["text"] for e in events if e["type"] == "delta"]
    assert "".join(deltas) == "回答开头 [1]，回答结尾 [1]。"
    assert events[-1]["type"] == "complete"
    assert events[-1]["answer"] == "回答开头 [1]，回答结尾 [1]。"
    assert events[-1]["verification"]["ok"] is True


def test_answer_stream_retrieval_only_yields_no_deltas(tmp_path):
    s, emb = seed(tmp_path, ["注意力机制的提出背景与动机。"])
    events = list(answer_stream(s, emb, None, "这篇论文讲了什么？", top=3))

    assert events[0]["retrieval_only"] is True
    assert [e["type"] for e in events] == ["context", "complete"]
    assert events[-1]["answer"] is None
    assert events[-1]["verification"] is None


def test_answer_stream_reports_citation_failure(tmp_path):
    s, emb = seed(tmp_path, ["注意力机制的提出背景与动机。"])
    llm = StreamFakeLLM(["没有引用的回答。"])
    events = list(answer_stream(s, emb, llm, "这篇论文讲了什么？", top=3))

    complete = events[-1]
    assert complete["verification"]["ok"] is False
    assert complete["verification"]["code"] == "numeric_citation_missing"


def test_answer_stream_empty_library_abstains(tmp_path):
    s = Store(tmp_path / "t.db")
    llm = StreamFakeLLM(["不应该被调用的回答 [1]。"])
    events = list(answer_stream(s, FakeEmbedder(), llm, "不存在的内容？", top=3))

    deltas = [e["text"] for e in events if e["type"] == "delta"]
    assert deltas == ["根据已有资料无法回答。"]
    complete = events[-1]
    assert complete["type"] == "complete"
    assert complete["answer"] == "根据已有资料无法回答。"
    assert complete["verification"] is None


# ---------- chat_turn 流式回调 ----------


def make_ctx(tmp_path):
    s = Store(tmp_path / "t.db")
    pid = s.upsert_paper(make_paper("a.pdf", title="注意力机制研究", year=2023))
    s.replace_chunks(
        pid,
        [Chunk(None, pid, 0, 1, "注意力机制是文本分类的关键技术。", FakeEmbedder.vecs_for("x"))],
    )
    return ToolContext(store=s, embedder=FakeEmbedder(), llm=None)


def test_chat_turn_streams_final_answer(tmp_path):
    ctx = make_ctx(tmp_path)
    llm = StreamingScriptLLM(
        [
            {
                "content": "先查看库状态。",
                "tool_calls": [{"id": "c1", "name": "library_status", "arguments": {}}],
            },
            {
                "content": "库里有 1 篇论文。",
                "tool_calls": [],
                "deltas": ["库里", "有 1 篇论文。"],
            },
        ]
    )
    ctx.llm = llm
    seen = []
    messages = [{"role": "user", "content": "库里有什么？"}]

    messages, logs = chat_turn(llm, messages, ctx, on_delta=seen.append)

    assert seen == ["库里", "有 1 篇论文。"]
    assert messages[-1]["role"] == "assistant"
    assert messages[-1]["content"] == "库里有 1 篇论文。"
    assert logs[-1].role == "assistant"


def test_chat_turn_ignores_on_delta_without_streaming_support(tmp_path):
    ctx = make_ctx(tmp_path)

    class PlainLLM:
        is_configured = True

        def __init__(self):
            self.calls = []

        def chat_with_tools(self, system, messages, tools):
            self.calls.append((system, messages, tools))
            return {"content": "库里有 1 篇论文。", "tool_calls": []}

    llm = PlainLLM()
    seen = []
    messages, _ = chat_turn(
        llm, [{"role": "user", "content": "q"}], ctx, on_delta=seen.append
    )

    assert seen == []
    assert len(llm.calls) == 1
    assert messages[-1]["content"] == "库里有 1 篇论文。"


# ---------- Web SSE 端点 ----------


def web_client(tmp_path, llm):
    s = Store(tmp_path / "t.db")
    pid = s.upsert_paper(make_paper("a.pdf", title="论文一", year=2023))
    s.replace_chunks(
        pid,
        [Chunk(None, pid, 0, 1, "注意力机制的提出背景与动机。", FakeEmbedder.vecs_for("x"))],
    )
    app = create_app(store=s, embedder=FakeEmbedder(), llm=llm)
    return FastAPITestClient(app, base_url="http://127.0.0.1")


def test_ask_stream_endpoint_emits_sse(tmp_path):
    client = web_client(tmp_path, StreamFakeLLM(["流式回答 [1]。"]))

    r = client.post("/api/ask/stream", json={"question": "这篇论文讲了什么？"})

    assert r.status_code == 200
    assert "text/event-stream" in r.headers["content-type"]
    assert '"type": "context"' in r.text
    assert '"type": "delta"' in r.text
    assert '"type": "complete"' in r.text
    assert "流式回答 [1]。" in r.text


def test_ask_stream_endpoint_mid_stream_error(tmp_path):
    class RaisingStreamLLM(StreamFakeLLM):
        def chat_stream(self, system, user):
            yield "开头"
            raise LLMError("模拟调用失败")

    client = web_client(tmp_path, RaisingStreamLLM([]))

    r = client.post("/api/ask/stream", json={"question": "这篇论文讲了什么？"})

    assert r.status_code == 200
    assert "开头" in r.text
    assert '"type": "error"' in r.text
    assert "模拟调用失败" in r.text


def test_ask_stream_endpoint_rejects_empty_question(tmp_path):
    client = web_client(tmp_path, StreamFakeLLM([]))

    r = client.post("/api/ask/stream", json={"question": "   "})

    assert r.status_code == 400


# ---------- CLI 流式展示 ----------


def _patch_cli_deps(monkeypatch):
    import pragent.cli as cli
    from pragent import config as config_mod

    monkeypatch.setattr(config_mod, "ensure_data_dir", lambda: None)
    monkeypatch.setattr(cli, "Store", lambda path: object())
    monkeypatch.setattr(cli, "Embedder", lambda model: object())
    monkeypatch.setattr(cli, "LLMClient", lambda *a, **k: SimpleNamespace(is_configured=True))


def test_cli_ask_streams_to_console(monkeypatch):
    import pragent.cli as cli

    _patch_cli_deps(monkeypatch)

    def fake_answer_stream(store, embedder, llm, question, top=8, web=False):
        yield {
            "type": "context",
            "sources": [
                {"n": 1, "title": "论文一", "year": 2023, "path": "a.pdf", "page": 1, "web": False}
            ],
            "hits": [],
            "web_papers": [],
            "retrieval_only": False,
        }
        yield {"type": "delta", "text": "回答第一部分"}
        yield {"type": "delta", "text": "[1]。"}
        yield {
            "type": "complete",
            "answer": "回答第一部分[1]。",
            "verification": {"ok": True, "code": "ok", "message": ""},
        }

    monkeypatch.setattr(cli, "answer_stream", fake_answer_stream)

    result = CliRunner().invoke(cli.app, ["ask", "问题"])

    assert result.exit_code == 0
    assert "回答第一部分[1]。" in result.stdout
    assert "  [1] 论文一（2023） 第1页 — a.pdf" in result.stdout


def test_cli_ask_stream_prints_verification_warning(monkeypatch):
    import pragent.cli as cli

    _patch_cli_deps(monkeypatch)

    def fake_answer_stream(store, embedder, llm, question, top=8, web=False):
        yield {
            "type": "context",
            "sources": [{"n": 1, "title": "论文一", "year": 2023, "path": "a.pdf", "page": 1, "web": False}],
            "hits": [],
            "web_papers": [],
            "retrieval_only": False,
        }
        yield {"type": "delta", "text": "没有引用。"}
        yield {
            "type": "complete",
            "answer": "没有引用。",
            "verification": {
                "ok": False,
                "code": "numeric_citation_missing",
                "message": "LLM 回答缺少 [n] 来源引用",
            },
        }

    monkeypatch.setattr(cli, "answer_stream", fake_answer_stream)

    result = CliRunner().invoke(cli.app, ["ask", "问题"])

    assert result.exit_code == 0
    assert "没有引用。" in result.stdout
    assert "引用验证失败" in result.stderr


def test_cli_ask_no_stream_uses_classic_path(monkeypatch):
    import pragent.cli as cli

    _patch_cli_deps(monkeypatch)
    monkeypatch.setattr(
        cli,
        "answer_ask",
        lambda *a, **k: (
            "[1] 一次性回答",
            [{"n": 1, "title": "论文一", "year": 2023, "path": "a.pdf", "page": 1, "web": False}],
            [],
            False,
            [],
        ),
    )

    result = CliRunner().invoke(cli.app, ["ask", "问题", "--no-stream"])

    assert result.exit_code == 0
    assert "[1] 一次性回答" in result.stdout
