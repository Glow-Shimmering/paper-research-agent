from importlib import resources

import pytest
from fastapi.testclient import TestClient as FastAPITestClient

from pragent import config
from pragent.llm import LLMError
from pragent.models import Chunk
from pragent.store import Store
from pragent.webapp import create_app, serve

from helpers import FakeEmbedder, make_paper


def TestClient(app, **kwargs):
    """Web tests use the same loopback origin as the production default."""
    kwargs.setdefault("base_url", "http://127.0.0.1")
    return FastAPITestClient(app, **kwargs)


def test_web_assets_are_installed_package_resources():
    web_dir = resources.files("pragent").joinpath("web")

    assert web_dir.is_dir()
    assert web_dir.joinpath("legacy", "index.html").is_file()
    assert web_dir.joinpath("legacy", "style.css").is_file()
    assert web_dir.joinpath("legacy", "app.js").is_file()
    assert web_dir.joinpath("templates", "projects.html").is_file()
    assert web_dir.joinpath("templates", "project_workspace.html").is_file()
    assert web_dir.joinpath("templates", "discover.html").is_file()
    assert web_dir.joinpath("templates", "library.html").is_file()
    assert web_dir.joinpath("templates", "fragments", "questions.html").is_file()
    assert web_dir.joinpath("templates", "fragments", "discovery_results.html").is_file()
    assert web_dir.joinpath("static", "app.css").is_file()
    assert web_dir.joinpath("static", "htmx.min.js").is_file()
    assert web_dir.joinpath("static", "HTMX-LICENSE.txt").is_file()


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


def test_api_key_protects_api_but_not_static_page(tmp_path):
    s = make_env(tmp_path)
    client = TestClient(
        create_app(store=s, embedder=FakeEmbedder(), llm=FakeLLM(), api_key="secret")
    )
    assert client.get("/").status_code == 200
    assert client.get("/api/status").status_code == 401
    assert client.get("/api/status", headers={"X-PRA-Key": "wrong"}).status_code == 401
    ok = client.get("/api/status", headers={"X-PRA-Key": "secret"})
    assert ok.status_code == 200


def test_unkeyed_api_rejects_non_loopback_host(tmp_path):
    s = make_env(tmp_path)
    client = FastAPITestClient(
        create_app(store=s, embedder=FakeEmbedder(), llm=FakeLLM(), api_key=""),
        base_url="http://pragent.example",
    )

    assert client.get("/").status_code == 200
    assert client.get("/api/status").status_code == 403


def test_api_rejects_cross_origin_browser_request(tmp_path):
    s = make_env(tmp_path)
    client = TestClient(create_app(store=s, embedder=FakeEmbedder(), llm=FakeLLM()))

    response = client.post(
        "/api/reindex",
        headers={"Origin": "https://evil.example"},
    )

    assert response.status_code == 403


def test_api_parameter_and_body_limits(tmp_path):
    s = make_env(tmp_path)
    client = TestClient(create_app(store=s, embedder=FakeEmbedder(), llm=FakeLLM(), api_key=""))
    assert client.get("/api/search", params={"q": "x", "top": 101}).status_code == 422
    assert client.get("/api/papers", params={"limit": 0}).status_code == 422
    response = client.post("/api/ask", json={"question": "x" * 20_001})
    assert response.status_code == 413


def test_api_streaming_body_limit_without_content_length(tmp_path):
    s = make_env(tmp_path)
    client = TestClient(create_app(store=s, embedder=FakeEmbedder(), llm=FakeLLM()))

    def oversized_chunks():
        yield b'{' + b'x' * 600_000
        yield b'x' * 600_000 + b'}'

    response = client.post(
        "/api/ask",
        content=oversized_chunks(),
        headers={"Content-Type": "application/json", "Transfer-Encoding": "chunked"},
    )

    assert response.status_code == 413


def test_remote_bind_requires_api_key(monkeypatch):
    monkeypatch.setattr(config, "WEB_API_KEY", "")
    with pytest.raises(RuntimeError, match="拒绝无鉴权"):
        serve(host="0.0.0.0", port=8000)


def test_remote_bind_rejects_plaintext_without_explicit_opt_in(monkeypatch):
    monkeypatch.setattr(config, "WEB_API_KEY", "secret")

    with pytest.raises(RuntimeError, match="明文 HTTP"):
        serve(host="0.0.0.0", port=8000)


def test_https_options_must_be_paired():
    with pytest.raises(RuntimeError, match="同时提供"):
        serve(host="127.0.0.1", port=8000, ssl_certfile="cert.pem")


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
    assert len(data["sources"]) == 2  # 命中片段 + 库藏目录
    assert data["sources"][0]["title"] == "论文一"
    assert data["sources"][0]["web"] is False
    assert data["sources"][1]["catalog"] is True
    assert len(data["hits"]) == 1
    assert data["web_papers"] == []


def test_ask_with_web(tmp_path, monkeypatch):
    import pragent.answer as answer_mod
    from pragent.websearch import WebPaper

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
    assert len(data["sources"]) == 3  # 片段 + 库藏 + 联网
    assert data["sources"][1]["catalog"] is True
    assert data["sources"][2]["web"] is True
    assert data["sources"][2]["path"] == "http://arxiv.org/abs/2501.1"
    assert len(data["web_papers"]) == 1
    assert data["web_papers"][0]["title"] == "Web Paper Title"


def test_ask_web_error_502(tmp_path, monkeypatch):
    import pragent.answer as answer_mod
    from pragent.websearch import WebSearchError

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
    assert "PRAgent" in r.text
    assert 'href="/ui/projects"' in r.text


def test_websearch_endpoint(monkeypatch, tmp_path):
    import pragent.webapp as webapp_mod
    from pragent.websearch import WebPaper

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
    import pragent.webapp as webapp_mod
    from pragent.websearch import WebSearchError

    def boom(q, limit):
        raise WebSearchError("arXiv 请求失败：超时")

    monkeypatch.setattr(webapp_mod, "search_papers", boom)
    s = make_env(tmp_path)
    client = TestClient(create_app(store=s, embedder=FakeEmbedder(), llm=FakeLLM()))
    r = client.get("/api/websearch", params={"q": "attention"})
    assert r.status_code == 502
    assert "arXiv 请求失败" in r.json()["detail"]
