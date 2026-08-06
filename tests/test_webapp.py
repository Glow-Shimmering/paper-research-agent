from fastapi.testclient import TestClient

from paper_agent.llm import LLMError
from paper_agent.models import Chunk
from paper_agent.store import Store
from paper_agent.webapp import create_app

from helpers import FakeEmbedder, make_paper


class FakeLLM:
    is_configured = True

    def __init__(self, reply="[1] 测试回答"):
        self.reply = reply

    def chat(self, system, user):
        return self.reply


class RaisingLLM(FakeLLM):
    def chat(self, system, user):
        raise LLMError("模拟调用失败")


def make_env(tmp_path, texts=None):
    s = Store(tmp_path / "t.db")
    if texts:
        pid = s.upsert_paper(make_paper("a.pdf", title="论文一", year=2023))
        s.replace_chunks(
            pid, [Chunk(None, pid, i, 1, t, FakeEmbedder.vecs_for(t)) for i, t in enumerate(texts)]
        )
    return s


def test_status(tmp_path):
    s = make_env(tmp_path)
    client = TestClient(create_app(store=s, embedder=FakeEmbedder(), llm=FakeLLM()))
    r = client.get("/api/status")
    assert r.status_code == 200
    data = r.json()
    assert data["papers"] == 0 and data["chunks"] == 0
    assert data["llm_configured"] is True
    assert data["embed_model"] == "fake"


def test_papers_list(tmp_path):
    s = make_env(tmp_path, ["正文内容。"])
    client = TestClient(create_app(store=s, embedder=FakeEmbedder(), llm=FakeLLM()))
    r = client.get("/api/papers")
    data = r.json()
    assert data["total"] == 1
    item = data["items"][0]
    assert item["title"] == "论文一"
    assert item["chunk_count"] == 1
    assert item["has_text"] is True
    assert item["authors"] == ["A"]


def test_search(tmp_path):
    s = make_env(tmp_path, ["注意力机制的动机与背景。"])
    client = TestClient(create_app(store=s, embedder=FakeEmbedder(), llm=FakeLLM()))
    r = client.get("/api/search", params={"q": "注意力"})
    assert r.status_code == 200
    hits = r.json()["hits"]
    assert hits and hits[0]["title"] == "论文一"
    assert "score" in hits[0] and "text" in hits[0]


def test_ask_ok(tmp_path):
    s = make_env(tmp_path, ["注意力机制内容。"])
    client = TestClient(create_app(store=s, embedder=FakeEmbedder(), llm=FakeLLM()))
    r = client.post("/api/ask", json={"question": "讲了什么？"})
    assert r.status_code == 200
    data = r.json()
    assert data["answer"] == "[1] 测试回答"
    assert data["retrieval_only"] is False
    assert len(data["sources"]) == 1
    assert data["sources"][0]["title"] == "论文一"
    assert data["sources"][0]["web"] is False
    assert len(data["hits"]) == 1
    assert data["web_papers"] == []


def test_ask_with_web(tmp_path, monkeypatch):
    import paper_agent.answer as answer_mod
    from paper_agent.websearch import WebPaper

    monkeypatch.setattr(
        answer_mod,
        "search_papers",
        lambda q, limit: [
            WebPaper(
                title="Web Paper Title", authors=["X"], year=2025,
                abstract="abs", url="http://arxiv.org/abs/2501.1", pdf_url=None,
            )
        ],
    )
    s = make_env(tmp_path, ["注意力机制内容。"])
    client = TestClient(create_app(store=s, embedder=FakeEmbedder(), llm=FakeLLM()))
    r = client.post("/api/ask", json={"question": "问题", "web": True})
    assert r.status_code == 200
    data = r.json()
    assert len(data["sources"]) == 2
    assert data["sources"][1]["web"] is True
    assert data["sources"][1]["path"] == "http://arxiv.org/abs/2501.1"
    assert len(data["web_papers"]) == 1
    assert data["web_papers"][0]["title"] == "Web Paper Title"


def test_ask_web_error_502(tmp_path, monkeypatch):
    import paper_agent.answer as answer_mod
    from paper_agent.websearch import WebSearchError

    def boom(q, limit):
        raise WebSearchError("arXiv 请求失败：超时")

    monkeypatch.setattr(answer_mod, "search_papers", boom)
    s = make_env(tmp_path, ["注意力机制内容。"])
    client = TestClient(create_app(store=s, embedder=FakeEmbedder(), llm=FakeLLM()))
    r = client.post("/api/ask", json={"question": "问题", "web": True})
    assert r.status_code == 502
    assert "arXiv 请求失败" in r.json()["detail"]


def test_ask_retrieval_only(tmp_path):
    s = make_env(tmp_path, ["内容。"])
    llm = FakeLLM()
    llm.is_configured = False
    client = TestClient(create_app(store=s, embedder=FakeEmbedder(), llm=llm))
    r = client.post("/api/ask", json={"question": "问题"})
    data = r.json()
    assert data["answer"] is None
    assert data["retrieval_only"] is True
    assert len(data["hits"]) == 1
    assert data["web_papers"] == []


def test_ask_empty_question_400(tmp_path):
    s = make_env(tmp_path)
    client = TestClient(create_app(store=s, embedder=FakeEmbedder(), llm=FakeLLM()))
    r = client.post("/api/ask", json={"question": "  "})
    assert r.status_code == 400


def test_ask_llm_error_500(tmp_path):
    s = make_env(tmp_path, ["内容。"])
    client = TestClient(create_app(store=s, embedder=FakeEmbedder(), llm=RaisingLLM()))
    r = client.post("/api/ask", json={"question": "问题"})
    assert r.status_code == 500
    assert "模拟调用失败" in r.json()["detail"]


def test_reindex_without_dir_400(tmp_path):
    s = make_env(tmp_path)
    client = TestClient(create_app(store=s, embedder=FakeEmbedder(), llm=FakeLLM()))
    r = client.post("/api/reindex")
    assert r.status_code == 400


def test_reindex(tmp_path):
    from helpers import make_pdf

    pdf_dir = tmp_path / "papers"
    pdf_dir.mkdir()
    make_pdf(pdf_dir / "a.pdf", ["内容。" * 30], {"title": "A"})
    s = Store(tmp_path / "t.db")
    s.meta_set("library_dir", str(pdf_dir))
    client = TestClient(create_app(store=s, embedder=FakeEmbedder(), llm=FakeLLM()))
    r = client.post("/api/reindex")
    assert r.status_code == 200
    data = r.json()
    assert data["added"] == 1
    assert s.stats() == (1, 1)


def test_index_page(tmp_path):
    s = make_env(tmp_path)
    client = TestClient(create_app(store=s, embedder=FakeEmbedder(), llm=FakeLLM()))
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "论文助手" in r.text


def test_websearch_endpoint(monkeypatch, tmp_path):
    import paper_agent.webapp as webapp_mod
    from paper_agent.websearch import WebPaper

    monkeypatch.setattr(
        webapp_mod,
        "search_papers",
        lambda q, limit: [
            WebPaper(
                title="Web Paper", authors=["A", "B"], year=2024,
                abstract="摘要内容", url="http://arxiv.org/abs/2401.1",
                pdf_url="http://arxiv.org/pdf/2401.1",
            )
        ],
    )
    s = make_env(tmp_path)
    client = TestClient(create_app(store=s, embedder=FakeEmbedder(), llm=FakeLLM()))
    r = client.get("/api/websearch", params={"q": "attention"})
    assert r.status_code == 200
    data = r.json()
    assert len(data["papers"]) == 1
    p = data["papers"][0]
    assert p["title"] == "Web Paper"
    assert p["pdf_url"] == "http://arxiv.org/pdf/2401.1"


def test_websearch_empty_400(tmp_path):
    s = make_env(tmp_path)
    client = TestClient(create_app(store=s, embedder=FakeEmbedder(), llm=FakeLLM()))
    assert client.get("/api/websearch", params={"q": "  "}).status_code == 400


def test_websearch_error_502(monkeypatch, tmp_path):
    import paper_agent.webapp as webapp_mod
    from paper_agent.websearch import WebSearchError

    def boom(q, limit):
        raise WebSearchError("arXiv 请求失败：超时")

    monkeypatch.setattr(webapp_mod, "search_papers", boom)
    s = make_env(tmp_path)
    client = TestClient(create_app(store=s, embedder=FakeEmbedder(), llm=FakeLLM()))
    r = client.get("/api/websearch", params={"q": "attention"})
    assert r.status_code == 502
    assert "arXiv 请求失败" in r.json()["detail"]
