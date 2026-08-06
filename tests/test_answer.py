import pytest

from paper_agent.answer import ask
from paper_agent.llm import LLMError, refine_metadata
from paper_agent.models import Chunk
from paper_agent.store import Store

from helpers import FakeEmbedder, make_paper


class FakeLLM:
    is_configured = True

    def __init__(self, reply="[1] 测试回答"):
        self.reply = reply
        self.last_system = None
        self.last_user = None

    def chat(self, system, user):
        self.last_system = system
        self.last_user = user
        return self.reply


class RaisingLLM(FakeLLM):
    def chat(self, system, user):
        raise LLMError("模拟调用失败")


def seed(tmp_path, texts: list[str], title="论文一", year=2023):
    s = Store(tmp_path / "t.db")
    pid = s.upsert_paper(make_paper("a.pdf", title=title, year=year))
    chunks = []
    for i, t in enumerate(texts):
        chunks.append(Chunk(None, pid, i, 1, t, FakeEmbedder.vecs_for(t)))
    s.replace_chunks(pid, chunks)
    return s, FakeEmbedder()


def test_ask_full_flow(tmp_path):
    s, emb = seed(tmp_path, ["注意力机制的提出背景与动机。"])
    llm = FakeLLM()
    answer, sources, hits, retrieval_only = ask(s, emb, llm, "这篇论文讲了什么？")
    assert answer == "[1] 测试回答"
    assert retrieval_only is False
    assert len(sources) == 1
    assert sources[0] == {"n": 1, "title": "论文一", "year": 2023, "path": "a.pdf", "page": 1}
    # prompt 模板：问题 + 带题目/年份/页码的引用块
    assert "问题：这篇论文讲了什么？" in llm.last_user
    assert "[1]《论文一》（2023）第1页：" in llm.last_user
    assert "注意力机制的提出背景与动机。" in llm.last_user
    assert "严谨的论文检索助手" in llm.last_system


def test_ask_year_page_omitted_when_missing(tmp_path):
    s, emb = seed(tmp_path, ["正文内容。"], title="无名", year=None)
    llm = FakeLLM()
    ask(s, emb, llm, "问题")
    assert "（None）" not in llm.last_user
    assert "[1]《无名》第1页：" in llm.last_user  # 年份省略，仅页码


def test_ask_retrieval_only_when_unconfigured(tmp_path):
    s, emb = seed(tmp_path, ["内容。"])
    llm = FakeLLM()
    llm.is_configured = False
    answer, sources, hits, retrieval_only = ask(s, emb, llm, "问题")
    assert answer is None and retrieval_only is True and sources == []
    assert len(hits) == 1


def test_ask_no_hits(tmp_path):
    s = Store(tmp_path / "t.db")
    llm = FakeLLM()
    answer, sources, hits, retrieval_only = ask(s, FakeEmbedder(), llm, "问题")
    assert answer == "根据已有资料无法回答。"
    assert sources == [] and hits == [] and retrieval_only is False


def test_ask_llm_error_raised(tmp_path):
    s, emb = seed(tmp_path, ["内容。"])
    with pytest.raises(LLMError):
        ask(s, emb, RaisingLLM(), "问题")


def test_refine_metadata_valid():
    llm = FakeLLM('{"title": "标题", "authors": ["甲", "乙"], "year": 2021}')
    assert refine_metadata(llm, "x.pdf", "text") == {"title": "标题", "authors": ["甲", "乙"], "year": 2021}


def test_refine_metadata_fenced_json():
    llm = FakeLLM('```json\n{"title": "标题"}\n```')
    assert refine_metadata(llm, "x.pdf", "text") == {"title": "标题"}


def test_refine_metadata_partial_and_invalid():
    # 部分字段非法 → 只保留合法字段
    llm = FakeLLM('{"title": "标题", "authors": "不是列表", "year": 99999}')
    assert refine_metadata(llm, "x.pdf", "text") == {"title": "标题"}
    # 非法 JSON / 非字典 → None
    llm = FakeLLM("不是 JSON")
    assert refine_metadata(llm, "x.pdf", "text") is None
    llm = FakeLLM("[1, 2, 3]")
    assert refine_metadata(llm, "x.pdf", "text") is None


def test_refine_metadata_llm_error():
    assert refine_metadata(RaisingLLM(), "x.pdf", "text") is None
