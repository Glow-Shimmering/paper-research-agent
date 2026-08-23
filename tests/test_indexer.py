from pathlib import Path

import numpy as np
import pytest

from pragent.indexer import index_library, index_pdf
from pragent.models import Chunk, Paper
from pragent.store import Store

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
    assert s.meta_get("library_dir") == str(pdf_dir.resolve())

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

    keep = pdf_dir / "keep.pdf"
    make_pdf(keep, ["保留的论文。"] * 30, {"title": "保留"})
    index_library(s, pdf_dir, FakeEmbedder(), progress=noop_progress)
    p.unlink()
    r = index_library(s, pdf_dir, FakeEmbedder(), progress=noop_progress)
    assert r["removed"] == 1
    assert s.stats()[0] == 1

    # no_prune 时保留已消失文件的条目
    make_pdf(p, ["又回来了。"] * 30, {"title": "回归"})
    index_library(s, pdf_dir, FakeEmbedder(), progress=noop_progress)
    p.unlink()
    r = index_library(s, pdf_dir, FakeEmbedder(), prune=False, progress=noop_progress)
    assert r["removed"] == 0
    assert s.stats()[0] == 2
    s.close()


def test_invalid_directory_rejected_before_touching_store(tmp_path):
    pdf_dir = tmp_path / "papers"
    pdf_dir.mkdir()
    make_pdf(pdf_dir / "a.pdf", ["原始内容。"] * 30)
    s = Store(tmp_path / "db.sqlite")
    index_library(s, pdf_dir, FakeEmbedder(), progress=noop_progress)
    before = s.stats()
    before_root = s.meta_get("library_dir")

    with pytest.raises(RuntimeError, match="不存在或无法访问"):
        index_library(s, tmp_path / "missing", FakeEmbedder(), progress=noop_progress)
    with pytest.raises(RuntimeError, match="不是文件夹"):
        index_library(s, pdf_dir / "a.pdf", FakeEmbedder(), progress=noop_progress)

    assert s.stats() == before
    assert s.meta_get("library_dir") == before_root
    s.close()


def test_empty_directory_does_not_prune_existing_library(tmp_path):
    pdf_dir = tmp_path / "papers"
    pdf_dir.mkdir()
    pdf = make_pdf(pdf_dir / "a.pdf", ["原始内容。"] * 30)
    s = Store(tmp_path / "db.sqlite")
    index_library(s, pdf_dir, FakeEmbedder(), progress=noop_progress)
    before = s.stats()
    pdf.unlink()

    with pytest.raises(RuntimeError, match="避免清空现有索引"):
        index_library(s, pdf_dir, FakeEmbedder(), progress=noop_progress)
    assert s.stats() == before

    result = index_library(s, pdf_dir, FakeEmbedder(), prune=False, progress=noop_progress)
    assert result["removed"] == 0
    assert s.stats() == before
    s.close()


def test_changed_root_requires_explicit_safe_mode(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    make_pdf(first / "a.pdf", ["第一篇。"] * 30)
    make_pdf(second / "b.pdf", ["第二篇。"] * 30)
    s = Store(tmp_path / "db.sqlite")
    index_library(s, first, FakeEmbedder(), progress=noop_progress)

    with pytest.raises(RuntimeError, match="--no-prune"):
        index_library(s, second, FakeEmbedder(), progress=noop_progress)
    assert s.stats()[0] == 1
    assert s.meta_get("library_dir") == str(first.resolve())

    result = index_library(s, second, FakeEmbedder(), prune=False, progress=noop_progress)
    assert result["added"] == 1
    assert s.stats()[0] == 2
    # 增量添加其他目录不能悄悄切换主目录，否则下次 prune 仍可能误删。
    assert s.meta_get("library_dir") == str(first.resolve())
    s.close()


def test_paths_are_stored_absolute(monkeypatch, tmp_path):
    pdf_dir = tmp_path / "papers"
    pdf_dir.mkdir()
    make_pdf(pdf_dir / "a.pdf", ["内容。"] * 30)
    monkeypatch.chdir(tmp_path)
    s = Store(tmp_path / "db.sqlite")

    index_library(s, Path("papers"), FakeEmbedder(), progress=noop_progress)

    paper = next(s.iter_papers())
    assert paper.path == str((pdf_dir / "a.pdf").resolve())
    assert s.meta_get("library_dir") == str(pdf_dir.resolve())
    s.close()


def test_force_failure_preserves_old_index(tmp_path):
    pdf_dir = tmp_path / "papers"
    pdf_dir.mkdir()
    pdf = make_pdf(pdf_dir / "a.pdf", ["仍需保留的旧索引内容。"] * 30)
    s = Store(tmp_path / "db.sqlite")
    index_library(s, pdf_dir, FakeEmbedder(model_name="m1"), progress=noop_progress)
    old_paper = next(s.iter_papers())
    old_stats = s.stats()
    pdf.write_bytes(b"not a pdf")

    with pytest.raises(RuntimeError, match="原索引未删除"):
        index_library(
            s,
            pdf_dir,
            FakeEmbedder(model_name="m2"),
            force=True,
            progress=noop_progress,
        )

    kept = next(s.iter_papers())
    assert kept.id == old_paper.id
    assert kept.sha256 == old_paper.sha256
    assert s.stats() == old_stats
    assert s.meta_get("embed_model") == "m1"
    s.close()


def test_force_database_failure_rolls_back_entire_replacement(monkeypatch, tmp_path):
    pdf_dir = tmp_path / "papers"
    pdf_dir.mkdir()
    pdf = make_pdf(pdf_dir / "a.pdf", ["旧内容。"] * 30)
    s = Store(tmp_path / "db.sqlite")
    index_library(s, pdf_dir, FakeEmbedder(model_name="m1"), progress=noop_progress)
    old_paper = next(s.iter_papers())
    old_stats = s.stats()
    make_pdf(pdf, ["可成功预处理的新内容。"] * 30)

    def fail_during_database_write(_paper_id, _chunks):
        raise RuntimeError("模拟 SQLite 写入失败")

    monkeypatch.setattr(s, "_replace_chunks_locked", fail_during_database_write)
    with pytest.raises(RuntimeError, match="模拟 SQLite 写入失败"):
        index_library(
            s,
            pdf_dir,
            FakeEmbedder(model_name="m2"),
            force=True,
            progress=noop_progress,
        )

    kept = next(s.iter_papers())
    assert kept.id == old_paper.id
    assert kept.sha256 == old_paper.sha256
    assert s.stats() == old_stats
    assert s.meta_get("embed_model") == "m1"
    s.close()


def test_incremental_database_failure_rolls_back_entire_batch(monkeypatch, tmp_path):
    pdf_dir = tmp_path / "papers"
    pdf_dir.mkdir()
    first_pdf = make_pdf(pdf_dir / "a.pdf", ["旧内容 A。"] * 30, {"title": "旧 A"})
    second_pdf = make_pdf(pdf_dir / "b.pdf", ["旧内容 B。"] * 30, {"title": "旧 B"})
    s = Store(tmp_path / "db.sqlite")
    index_library(s, pdf_dir, FakeEmbedder(), progress=noop_progress)
    before = {paper.path: (paper.sha256, paper.title) for paper in s.iter_papers()}
    before_stats = s.stats()
    before_revision = s.revision
    before_model = s.meta_get("embed_model")
    before_root = s.meta_get("library_dir")

    make_pdf(first_pdf, ["新内容 A。"] * 30, {"title": "新 A"})
    make_pdf(second_pdf, ["新内容 B。"] * 30, {"title": "新 B"})
    original_replace = s._replace_chunks_locked
    write_count = 0

    def fail_on_second_paper(paper_id, chunks):
        nonlocal write_count
        write_count += 1
        if write_count == 2:
            raise RuntimeError("模拟第二篇 SQLite 写入失败")
        return original_replace(paper_id, chunks)

    monkeypatch.setattr(s, "_replace_chunks_locked", fail_on_second_paper)
    with pytest.raises(RuntimeError, match="第二篇 SQLite 写入失败"):
        index_library(s, pdf_dir, FakeEmbedder(), progress=noop_progress)

    after = {paper.path: (paper.sha256, paper.title) for paper in s.iter_papers()}
    assert after == before
    assert s.stats() == before_stats
    assert s.revision == before_revision
    assert s.meta_get("embed_model") == before_model
    assert s.meta_get("library_dir") == before_root
    s.close()


def test_missing_embed_model_provenance_requires_force(tmp_path):
    pdf_dir = tmp_path / "papers"
    pdf_dir.mkdir()
    pdf = make_pdf(pdf_dir / "a.pdf", ["正文内容。"] * 30)
    s = Store(tmp_path / "db.sqlite")
    paper = Paper(
        id=None,
        path=str(pdf.resolve()),
        sha256="unknown-provenance",
        title="旧论文",
        authors=[],
        year=None,
        page_count=1,
        has_text=True,
        indexed_at="2026-01-01T00:00:00",
    )
    s.upsert_paper(
        paper,
        [Chunk(None, 0, 0, 1, "旧向量", np.ones(8, dtype=np.float32))],
    )
    before_revision = s.revision

    with pytest.raises(RuntimeError, match="缺少嵌入模型来源.*--force"):
        index_library(s, pdf_dir, FakeEmbedder(), progress=noop_progress)

    assert s.revision == before_revision
    assert s.meta_get("embed_model") is None
    result = index_library(s, pdf_dir, FakeEmbedder(), force=True, progress=noop_progress)
    assert result["added"] == 1
    assert s.meta_get("embed_model") == "fake"
    s.close()


def test_index_pdf_only_touches_requested_file_and_never_prunes(tmp_path):
    library = tmp_path / "library"
    downloads = tmp_path / "downloads"
    library.mkdir()
    downloads.mkdir()
    make_pdf(library / "main.pdf", ["主目录论文。"] * 30)
    downloaded = make_pdf(downloads / "new.pdf", ["新下载论文。"] * 30)
    untouched = make_pdf(downloads / "untouched.pdf", ["不应被扫描。"] * 30)
    s = Store(tmp_path / "db.sqlite")
    index_library(s, library, FakeEmbedder(), progress=noop_progress)

    result = index_pdf(s, downloaded, FakeEmbedder(), progress=noop_progress)
    assert result["added"] == 1
    assert s.paper_by_path(str(downloaded.resolve())) is not None
    assert s.paper_by_path(str(untouched.resolve())) is None
    assert s.paper_by_path(str((library / "main.pdf").resolve())) is not None

    # 主目录后续 prune 也只清理主目录范围，保留单文件增量来源。
    index_library(s, library, FakeEmbedder(), progress=noop_progress)
    assert s.stats()[0] == 2
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

    import pragent.indexer as indexer_mod

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

    import pragent.indexer as indexer_mod

    monkeypatch.setattr(indexer_mod, "refine_metadata", fake_refine)
    s = Store(tmp_path / "db.sqlite")
    index_library(s, pdf_dir, FakeEmbedder(), refine=True, llm=FakeLLM(), progress=noop_progress)
    paper = s.paper_by_path(str(pdf_dir / "no_meta.pdf"))
    assert paper.title == "no meta"  # 文件名兜底（_ 规范化为空格）
    s.close()
