import pytest

from paper_agent.store import Store
from paper_agent.tools import TOOLS, ToolContext, execute_tool

from helpers import FakeEmbedder, make_paper


class FakeLLM:
    is_configured = True


def make_ctx(tmp_path, library_dir=None):
    s = Store(tmp_path / "t.db")
    pid = s.upsert_paper(make_paper("a.pdf", title="注意力机制研究", authors=["张三"], year=2023))
    from paper_agent.models import Chunk
    from helpers import FakeEmbedder as FE

    s.replace_chunks(
        pid,
        [
            Chunk(None, pid, 0, 1, "注意力机制是文本分类的关键技术。", FE.vecs_for("注意力机制是文本分类的关键技术。")),
            Chunk(None, pid, 1, 2, "Transformer 使用自注意力。", FE.vecs_for("Transformer 使用自注意力。")),
        ],
    )
    if library_dir is not None:
        s.meta_set("library_dir", str(library_dir))
    return ToolContext(store=s, embedder=FakeEmbedder(), llm=FakeLLM())


def test_tools_schema_complete():
    from paper_agent.tools import SCHEMA_NAMES

    names = [t["function"]["name"] for t in TOOLS]
    assert set(names) == SCHEMA_NAMES == {
        "local_search", "web_search", "download_paper", "index_papers", "list_papers", "library_status",
    }
    for t in TOOLS:
        assert t["type"] == "function"
        assert "description" in t["function"]
        assert "parameters" in t["function"]


def test_local_search(tmp_path):
    from paper_agent.tools import ToolContext

    ctx = make_ctx(tmp_path)
    out = execute_tool("local_search", {"query": "注意力机制"}, ctx)
    assert "注意力机制研究" in out
    assert "第1页" in out or "page" in out


def test_local_search_no_hits(tmp_path):
    from paper_agent.tools import ToolContext

    s = Store(tmp_path / "t.db")  # 空库：混合检索无命中
    ctx = ToolContext(store=s, embedder=FakeEmbedder(), llm=FakeLLM())
    out = execute_tool("local_search", {"query": "任意词"}, ctx)
    assert "未找到" in out


def test_web_search(monkeypatch, tmp_path):
    from paper_agent import websearch as ws_mod
    from paper_agent.tools import ToolContext

    monkeypatch.setattr(
        ws_mod,
        "search_papers",
        lambda q, limit: [
            ws_mod.WebPaper(
                title="A Survey", authors=["A"], year=2025, abstract="abs",
                url="http://arxiv.org/abs/2501.1", pdf_url=None,
            )
        ],
    )
    ctx = make_ctx(tmp_path)
    out = execute_tool("web_search", {"query": "llm survey"}, ctx)
    assert "A Survey" in out and "2501.1" in out


def test_web_search_failure(monkeypatch, tmp_path):
    from paper_agent import websearch as ws_mod
    from paper_agent.tools import ToolContext

    def boom(q, limit):
        raise ws_mod.WebSearchError("超时")

    monkeypatch.setattr(ws_mod, "search_papers", boom)
    ctx = make_ctx(tmp_path)
    out = execute_tool("web_search", {"query": "x"}, ctx)
    assert "联网检索失败" in out


def test_download_paper(monkeypatch, tmp_path):
    from paper_agent import download as dl_mod
    from paper_agent.tools import ToolContext

    pdf_bytes = b"%PDF-1.7 fake " * 50

    def fake_download(url, target_dir, timeout=60):
        target = target_dir / "2402.11651.pdf"
        target.write_bytes(pdf_bytes)
        return target

    monkeypatch.setattr(dl_mod, "download_pdf", fake_download)
    monkeypatch.setattr(
        "paper_agent.indexer.index_library",
        lambda store, d, embedder, **kw: {"added": 1, "updated": 0, "unchanged": 0, "failed": 0},
    )
    monkeypatch.setattr("paper_agent.config.download_dir_override", lambda: None)
    ctx = make_ctx(tmp_path, library_dir=tmp_path / "lib")
    (tmp_path / "lib").mkdir()
    out = execute_tool("download_paper", {"url": "https://arxiv.org/abs/2402.11651"}, ctx)
    assert "已下载并索引" in out and "2402.11651" in out
    assert (tmp_path / "lib" / "2402.11651.pdf").exists()


def test_download_paper_override_dir_priority(monkeypatch, tmp_path):
    """显式配置目录优先于论文库目录。"""
    from paper_agent import download as dl_mod
    from paper_agent.tools import ToolContext

    override = tmp_path / "override"
    pdf_bytes = b"%PDF-1.7 fake " * 50

    def fake_download(url, target_dir, timeout=60):
        target = target_dir / "2402.11651.pdf"
        target.write_bytes(pdf_bytes)
        return target

    monkeypatch.setattr(dl_mod, "download_pdf", fake_download)
    monkeypatch.setattr(
        "paper_agent.indexer.index_library",
        lambda store, d, embedder, **kw: {"added": 1, "updated": 0, "unchanged": 0, "failed": 0},
    )
    monkeypatch.setattr("paper_agent.config.download_dir_override", lambda: override)
    lib = tmp_path / "lib"
    lib.mkdir()
    ctx = make_ctx(tmp_path, library_dir=lib)
    out = execute_tool("download_paper", {"url": "https://arxiv.org/abs/2402.11651"}, ctx)
    assert (override / "2402.11651.pdf").exists()
    assert not (lib / "2402.11651.pdf").exists()


def test_download_paper_no_library(monkeypatch, tmp_path):
    monkeypatch.setattr("paper_agent.config.download_dir_override", lambda: None)
    ctx = make_ctx(tmp_path)  # 无 library_dir
    out = execute_tool("download_paper", {"url": "https://arxiv.org/abs/2402.11651"}, ctx)
    assert "未配置下载目录" in out and "PAPER_DOWNLOAD_DIR" in out


def test_list_papers_and_status(tmp_path):
    ctx = make_ctx(tmp_path)
    out = execute_tool("list_papers", {}, ctx)
    assert "共 1 篇" in out and "注意力机制研究" in out
    out = execute_tool("library_status", {}, ctx)
    assert "论文 1 篇" in out


def test_unknown_tool(tmp_path):
    ctx = make_ctx(tmp_path)
    out = execute_tool("no_such_tool", {}, ctx)
    assert "未知工具" in out and "local_search" in out


def test_tool_error_returns_text(tmp_path):
    ctx = make_ctx(tmp_path)
    out = execute_tool("local_search", {"query": 123}, ctx)  # 参数类型错误
    assert "工具" in out and ("参数" in out or "失败" in out)
