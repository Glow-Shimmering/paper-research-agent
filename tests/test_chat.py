import pytest

from paper_agent.chat import MAX_TOOL_ROUNDS, chat_turn
from paper_agent.models import Chunk
from paper_agent.store import Store
from paper_agent.tools import ToolContext

from helpers import FakeEmbedder, make_paper


class FakeLLM:
    """按脚本返回响应：content + tool_calls。"""

    is_configured = True

    def __init__(self, script):
        self.script = list(script)  # 每次 chat_with_tools 消费一个
        self.calls = []

    def chat_with_tools(self, system, messages, tools):
        self.calls.append((system, list(messages), list(tools)))
        return self.script.pop(0)


def make_ctx(tmp_path):
    s = Store(tmp_path / "t.db")
    pid = s.upsert_paper(make_paper("a.pdf", title="注意力机制研究", year=2023))
    s.replace_chunks(pid, [Chunk(None, pid, 0, 1, "注意力机制是文本分类的关键技术。", FakeEmbedder.vecs_for("x"))])
    return ToolContext(store=s, embedder=FakeEmbedder(), llm=FakeLLM([]))


def tool_call(cid, name, args):
    return {"id": cid, "name": name, "arguments": args}


def test_single_turn_no_tools(tmp_path):
    llm = FakeLLM([{"content": "你好！", "tool_calls": []}])
    ctx = make_ctx(tmp_path)
    messages, logs = chat_turn(llm, [{"role": "user", "content": "hi"}], ctx)
    assert messages[-1] == {"role": "assistant", "content": "你好！", "tool_calls": []}
    assert [l.role for l in logs] == ["assistant"]
    assert len(llm.calls) == 1


def test_tool_call_roundtrip(tmp_path):
    script = [
        {
            "content": None,
            "tool_calls": [tool_call("c1", "library_status", {})],
        },
        {"content": "库里有 1 篇论文。", "tool_calls": []},
    ]
    llm = FakeLLM(script)
    ctx = make_ctx(tmp_path)
    messages, logs = chat_turn(llm, [{"role": "user", "content": "库里有什么？"}], ctx)

    # assistant 消息保留 tool_calls（OpenAI 协议要求）
    assistant_msg = messages[1]
    assert assistant_msg["role"] == "assistant"
    assert assistant_msg["tool_calls"][0]["function"]["name"] == "library_status"
    assert assistant_msg["tool_calls"][0]["id"] == "c1"
    # tool 结果消息
    tool_msg = messages[2]
    assert tool_msg["role"] == "tool"
    assert tool_msg["tool_call_id"] == "c1"
    assert "论文 1 篇" in tool_msg["content"]
    # 最终回答
    assert messages[3]["content"] == "库里有 1 篇论文。"
    # 日志
    assert [l.role for l in logs] == ["assistant", "tool", "assistant"]
    assert logs[1].tool_name == "library_status"
    assert "论文 1 篇" in logs[1].tool_result
    # 第二轮请求带上了历史（含 tool 消息）
    assert len(llm.calls) == 2
    roles = [m["role"] for m in llm.calls[1][1]]
    assert roles == ["user", "assistant", "tool"]


def test_tool_error_result_to_llm(tmp_path):
    script = [
        {"content": None, "tool_calls": [tool_call("c1", "no_such_tool", {})]},
        {"content": "完成", "tool_calls": []},
    ]
    llm = FakeLLM(script)
    ctx = make_ctx(tmp_path)
    messages, logs = chat_turn(llm, [{"role": "user", "content": "x"}], ctx)
    assert "未知工具" in messages[2]["content"]
    assert "未知工具" in logs[1].tool_result


def test_max_rounds_cutoff(tmp_path):
    # 模型始终要求调用工具 → 循环在 MAX_TOOL_ROUNDS 轮后停止
    script = [{"content": None, "tool_calls": [tool_call(f"c{i}", "library_status", {})]} for i in range(MAX_TOOL_ROUNDS + 3)]
    llm = FakeLLM(script)
    ctx = make_ctx(tmp_path)
    messages, logs = chat_turn(llm, [{"role": "user", "content": "x"}], ctx)
    assert len(llm.calls) == MAX_TOOL_ROUNDS
    assert logs[-1].role == "error"
    assert "轮次过多" in logs[-1].content


def test_local_search_tool_in_dialogue(tmp_path):
    script = [
        {"content": None, "tool_calls": [tool_call("c1", "local_search", {"query": "注意力"})]},
        {"content": "本地库有相关内容。", "tool_calls": []},
    ]
    llm = FakeLLM(script)
    ctx = make_ctx(tmp_path)
    messages, logs = chat_turn(llm, [{"role": "user", "content": "查一下"}], ctx)
    assert "注意力机制研究" in messages[2]["content"]
