import pytest

from paper_agent.indexer import index_library
from paper_agent.store import Store

from helpers import FakeEmbedder, make_pdf, noop_progress


def test_index_add_and_unchanged(tmp_path):
    pdf_dir = tmp_path / "papers"
    pdf_dir.mkdir()
    make_pdf(pdf_dir / "a.pdf", ["第一页正文内容，研究文本分类。"] * 30, {"title": "论文A", "author": "张三"})
    make_pdf(pdf_dir / "b.pdf", ["Second page body about transformer models."] * 30, {"title": "Paper B", "author": "Alice"})

    s = Store(tmp_path / "db.sqlite")
    r1 = index_library(s, pdf_dir, FakeEmbedder(), progress=noop_progress)
    assert r1["added"] == 2
    papers, chunks = s.stats()
    assert papers == 2 and chunks > 0
    assert s.meta_get("embed_model") == "fake"
    assert s.meta_get("library_dir") == str(pdf_dir)

    r2 = index_library(s, pdf_dir, FakeEmbedder(), progress=noop_progress)
    assert r2["unchanged"] == 2 and r2["added"] == 0
    assert s.stats() == (2, chunks)
    s.close()


def test_updated_and_prune(tmp_path):
    pdf_dir = tmp_path / "papers"
    pdf_dir.mkdir()
    p = pdf_dir / "a.pdf"
    make_pdf(p, ["第一版内容。"] * 30, {"title": "论文A"})

    s = Store(tmp_path / "db.sqlite")
    index_library(s, pdf_dir, FakeEmbedder(), progress=noop_progress)

    make_pdf(p, ["第二版完全不同的内容。"] * 30, {"title": "论文A改"})
    r = index_library(s, pdf_dir, FakeEmbedder(), progress=noop_progress)
    assert r["updated"] == 1
    paper = s.paper_by_path(str(p))
    assert paper.title == "论文A改"
    assert paper.sha256 != "s"

    p.unlink()
    r = index_library(s, pdf_dir, FakeEmbedder(), progress=noop_progress)
    assert r["removed"] == 1
    assert s.stats() == (0, 0)

    # no_prune 时保留已消失文件的条目
    make_pdf(p, ["又回来了。"] * 30, {"title": "回归"})
    index_library(s, pdf_dir, FakeEmbedder(), progress=noop_progress)
    p.unlink()
    r = index_library(s, pdf_dir, FakeEmbedder(), prune=False, progress=noop_progress)
    assert r["removed"] == 0
    assert s.stats()[0] == 1
    s.close()


def test_model_mismatch_requires_force(tmp_path):
    pdf_dir = tmp_path / "papers"
    pdf_dir.mkdir()
    make_pdf(pdf_dir / "a.pdf", ["内容。"] * 30)

    s = Store(tmp_path / "db.sqlite")
    index_library(s, pdf_dir, FakeEmbedder(model_name="m1"), progress=noop_progress)

    with pytest.raises(RuntimeError, match="--force"):
        index_library(s, pdf_dir, FakeEmbedder(model_name="m2"), progress=noop_progress)

    r = index_library(s, pdf_dir, FakeEmbedder(model_name="m2"), force=True, progress=noop_progress)
    assert r["added"] == 1
    assert s.meta_get("embed_model") == "m2"
    s.close()


def test_broken_file_failed(tmp_path):
    pdf_dir = tmp_path / "papers"
    pdf_dir.mkdir()
    (pdf_dir / "bad.pdf").write_bytes(b"not a pdf at all")
    s = Store(tmp_path / "db.sqlite")
    r = index_library(s, pdf_dir, FakeEmbedder(), progress=noop_progress)
    assert r["failed"] == 1
    assert s.stats() == (0, 0)
    s.close()


def test_no_text_skipped(tmp_path):
    pdf_dir = tmp_path / "papers"
    pdf_dir.mkdir()
    make_pdf(pdf_dir / "scan.pdf", [""])
    s = Store(tmp_path / "db.sqlite")
    r = index_library(s, pdf_dir, FakeEmbedder(), progress=noop_progress)
    assert r["skipped_no_text"] == 1
    paper = s.paper_by_path(str(pdf_dir / "scan.pdf"))
    assert paper is not None and paper.has_text is False
    # 二次运行：无文本论文应被跳过（unchanged）
    r2 = index_library(s, pdf_dir, FakeEmbedder(), progress=noop_progress)
    assert r2["unchanged"] == 1
    s.close()


def test_refine_metadata_used(monkeypatch, tmp_path):
    pdf_dir = tmp_path / "papers"
    pdf_dir.mkdir()
    make_pdf(pdf_dir / "no_meta.pdf", ["一些正文内容。"] * 30)  # 无内嵌元数据 → 标题=文件名

    class FakeLLM:
        is_configured = True

    def fake_refine(llm, filename, first_text):
        return {"title": "提炼出的标题", "authors": ["李四"], "year": 2024}

    import paper_agent.indexer as indexer_mod

    monkeypatch.setattr(indexer_mod, "refine_metadata", fake_refine)
    s = Store(tmp_path / "db.sqlite")
    index_library(s, pdf_dir, FakeEmbedder(), refine=True, llm=FakeLLM(), progress=noop_progress)
    paper = s.paper_by_path(str(pdf_dir / "no_meta.pdf"))
    assert paper.title == "提炼出的标题"
    assert paper.authors == ["李四"]
    assert paper.year == 2024
    s.close()


def test_refine_failure_keeps_fallback(monkeypatch, tmp_path):
    pdf_dir = tmp_path / "papers"
    pdf_dir.mkdir()
    make_pdf(pdf_dir / "no_meta.pdf", ["一些正文内容。"] * 30)

    class FakeLLM:
        is_configured = True

    def fake_refine(llm, filename, first_text):
        return None  # 解析失败

    import paper_agent.indexer as indexer_mod

    monkeypatch.setattr(indexer_mod, "refine_metadata", fake_refine)
    s = Store(tmp_path / "db.sqlite")
    index_library(s, pdf_dir, FakeEmbedder(), refine=True, llm=FakeLLM(), progress=noop_progress)
    paper = s.paper_by_path(str(pdf_dir / "no_meta.pdf"))
    assert paper.title == "no meta"  # 文件名兜底（_ 规范化为空格）
    s.close()
