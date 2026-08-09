import pytest

from paper_agent.answer import ask
from paper_agent.llm import LLMError, refine_metadata
from paper_agent.models import Chunk
from paper_agent.store import Store
from paper_agent.websearch import WebPaper, WebSearchError

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


def fake_web_papers():
    return [
        WebPaper(
            title="A New Attention Architecture",
            authors=["Carol", "Dave"],
            year=2025,
            abstract="We present a novel attention architecture for sequence modeling.",
            url="http://arxiv.org/abs/2501.00001",
            pdf_url="http://arxiv.org/pdf/2501.00001",
        )
    ]


def test_ask_full_flow(tmp_path):
    s, emb = seed(tmp_path, ["注意力机制的提出背景与动机。"])
    llm = FakeLLM()
    answer, sources, hits, retrieval_only, web_papers = ask(s, emb, llm, "这篇论文讲了什么？")
    assert answer == "[1] 测试回答"
    assert retrieval_only is False
    assert web_papers == []
    assert len(sources) == 1
    assert sources[0] == {
        "n": 1, "title": "论文一", "year": 2023, "path": "a.pdf", "page": 1, "web": False,
    }
    # prompt 模板：问题 + 带题目/年份/页码的引用块
    assert "问题：这篇论文讲了什么？" in llm.last_user
    assert "[1]《论文一》（2023）第1页：" in llm.last_user
    assert "注意力机制的提出背景与动机。" in llm.last_user
    assert "严谨的论文检索助手" in llm.last_system


def test_ask_web_flow(tmp_path, monkeypatch):
    import paper_agent.answer as answer_mod

    monkeypatch.setattr(answer_mod, "search_papers", lambda q, limit: fake_web_papers())
    s, emb = seed(tmp_path, ["注意力机制的提出背景与动机。"])
    llm = FakeLLM()
    answer, sources, hits, retrieval_only, web_papers = ask(s, emb, llm, "最新的注意力架构", web=True)
    assert retrieval_only is False
    assert len(web_papers) == 1
    # 本地 1 条 + 联网 1 条，编号连续
    assert "[1]《论文一》（2023）第1页：" in llm.last_user
    assert "[2]《A New Attention Architecture》（2025）[arXiv 联网] Carol、Dave：" in llm.last_user
    assert "novel attention architecture" in llm.last_user
    assert sources[1] == {
        "n": 2, "title": "A New Attention Architecture", "year": 2025,
        "path": "http://arxiv.org/abs/2501.00001", "page": None, "web": True,
    }


def test_ask_web_only_no_local_hits(tmp_path, monkeypatch):
    import paper_agent.answer as answer_mod

    monkeypatch.setattr(answer_mod, "search_papers", lambda q, limit: fake_web_papers())
    s = Store(tmp_path / "t.db")  # 空库
    llm = FakeLLM()
    answer, sources, hits, retrieval_only, web_papers = ask(s, FakeEmbedder(), llm, "新方法", web=True)
    assert answer == "[1] 测试回答"
    assert hits == [] and len(web_papers) == 1
    assert "[1]《A New Attention Architecture》（2025）[arXiv 联网]" in llm.last_user


def test_ask_web_error_raised(tmp_path, monkeypatch):
    import paper_agent.answer as answer_mod

    def boom(q, limit):
        raise WebSearchError("arXiv 请求失败：超时")

    monkeypatch.setattr(answer_mod, "search_papers", boom)
    s, emb = seed(tmp_path, ["内容。"])
    with pytest.raises(WebSearchError, match="arXiv 请求失败"):
        ask(s, emb, FakeLLM(), "问题", web=True)


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
    answer, sources, hits, retrieval_only, web_papers = ask(s, emb, llm, "问题")
    assert answer is None and retrieval_only is True and sources == []
    assert len(hits) == 1


def test_ask_retrieval_only_with_web(tmp_path, monkeypatch):
    import paper_agent.answer as answer_mod

    monkeypatch.setattr(answer_mod, "search_papers", lambda q, limit: fake_web_papers())
    s, emb = seed(tmp_path, ["内容。"])
    llm = FakeLLM()
    llm.is_configured = False
    answer, sources, hits, retrieval_only, web_papers = ask(s, emb, llm, "问题", web=True)
    assert answer is None and retrieval_only is True
    assert len(hits) == 1 and len(web_papers) == 1


def test_ask_no_hits(tmp_path):
    s = Store(tmp_path / "t.db")
    llm = FakeLLM()
    answer, sources, hits, retrieval_only, web_papers = ask(s, FakeEmbedder(), llm, "问题")
    assert answer == "根据已有资料无法回答。"
    assert sources == [] and hits == [] and retrieval_only is False and web_papers == []


def test_ask_llm_error_raised(tmp_path):
    s, emb = seed(tmp_path, ["内容。"])
    with pytest.raises(LLMError):
        ask(s, emb, RaisingLLM(), "问题")


def test_ask_rejects_invalid_citation(tmp_path):
    s, emb = seed(tmp_path, ["内容。"])
    with pytest.raises(LLMError, match="不存在的来源引用"):
        ask(s, emb, FakeLLM("结论 [2]"), "问题")


def test_ask_rejects_missing_citation_but_allows_abstention(tmp_path):
    s, emb = seed(tmp_path, ["内容。"])
    with pytest.raises(LLMError, match="缺少.*来源引用"):
        ask(s, emb, FakeLLM("这是没有引用的结论"), "问题")
    answer, *_ = ask(s, emb, FakeLLM("根据已有资料无法回答。"), "问题")
    assert answer == "根据已有资料无法回答。"


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
