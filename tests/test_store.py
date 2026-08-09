import numpy as np
import pytest

from paper_agent.models import Chunk, Paper
from paper_agent.store import RevisionConflictError, Store


def make_paper(path="a.pdf", **kw):
    base = dict(
        id=None,
        path=path,
        sha256="sha1",
        title="标题",
        authors=["张三"],
        year=2020,
        page_count=1,
        has_text=True,
        indexed_at="2026-01-01T00:00:00",
    )
    base.update(kw)
    return Paper(**base)


def test_upsert_idempotent(tmp_path):
    s = Store(tmp_path / "t.db")
    id1 = s.upsert_paper(make_paper())
    id2 = s.upsert_paper(make_paper(title="新标题"))
    assert id1 == id2
    total, papers = s.list_papers(None, 50, 0)
    assert total == 1
    assert papers[0].title == "新标题"
    assert papers[0].authors == ["张三"]
    s.close()


def test_replace_chunks(tmp_path):
    s = Store(tmp_path / "t.db")
    pid = s.upsert_paper(make_paper())
    s.replace_chunks(
        pid,
        [
            Chunk(None, pid, 0, 1, "hello", np.ones(4, dtype=np.float32)),
            Chunk(None, pid, 1, 2, "world"),
        ],
    )
    ch = s.get_chunks_by_paper(pid)
    assert [c.seq for c in ch] == [0, 1]
    s.replace_chunks(pid, [Chunk(None, pid, 0, 3, "new")])
    ch = s.get_chunks_by_paper(pid)
    assert len(ch) == 1 and ch[0].page == 3 and ch[0].text == "new"
    s.close()


def test_upsert_with_chunks_single_tx(tmp_path):
    s = Store(tmp_path / "t.db")
    pid = s.upsert_paper(
        make_paper(),
        chunks=[Chunk(None, 0, 0, 1, "x", np.zeros(2, dtype=np.float32))],
    )
    assert len(s.get_chunks_by_paper(pid)) == 1
    s.close()


def test_embedding_roundtrip(tmp_path):
    s = Store(tmp_path / "t.db")
    pid = s.upsert_paper(make_paper())
    arr = np.random.default_rng(42).random((4, 8), dtype=np.float32)
    s.replace_chunks(pid, [Chunk(None, pid, i, 1, f"c{i}", arr[i]) for i in range(4)])
    matrix, ids = s.all_embeddings()
    assert matrix.shape == (4, 8)
    assert ids == [1, 2, 3, 4]
    np.testing.assert_allclose(matrix, arr)
    s.close()


def test_all_chunks_alignment(tmp_path):
    # all_chunks 与 all_embeddings 同序：按 id 升序、仅含非空嵌入
    s = Store(tmp_path / "t.db")
    pid = s.upsert_paper(make_paper())
    s.replace_chunks(
        pid,
        [
            Chunk(None, pid, 0, 1, "有嵌入", np.zeros(2, dtype=np.float32)),
            Chunk(None, pid, 1, 2, "无嵌入"),
            Chunk(None, pid, 2, 3, "有嵌入2", np.ones(2, dtype=np.float32)),
        ],
    )
    chunks = s.all_chunks()
    _, ids = s.all_embeddings()
    assert [c.id for c in chunks] == ids == [1, 3]
    assert [c.text for c in chunks] == ["有嵌入", "有嵌入2"]
    s.close()


def test_search_snapshot_is_aligned_and_self_contained(tmp_path):
    s = Store(tmp_path / "t.db")
    pid = s.upsert_paper(make_paper(title="快照论文", authors=["甲", "乙"]))
    s.replace_chunks(
        pid,
        [
            Chunk(None, pid, 0, 1, "第一块", np.array([1, 0], dtype=np.float32)),
            Chunk(None, pid, 1, 2, "无向量"),
            Chunk(None, pid, 2, 3, "第三块", np.array([0, 1], dtype=np.float32)),
        ],
    )
    s.meta_set("embed_model", "m1")

    snapshot = s.search_snapshot()

    assert [item.text for item in snapshot.items] == ["第一块", "第三块"]
    assert [item.chunk_id for item in snapshot.items] == [1, 3]
    assert snapshot.items[0].title == "快照论文"
    assert snapshot.items[0].authors == ("甲", "乙")
    np.testing.assert_array_equal(
        snapshot.embeddings,
        np.array([[1, 0], [0, 1]], dtype=np.float32),
    )
    assert snapshot.embed_model == "m1"
    assert snapshot.revision == s.revision
    assert not snapshot.embeddings.flags.writeable
    s.close()


def test_revision_invalidates_search_visible_mutations(tmp_path):
    s = Store(tmp_path / "t.db")
    assert s.revision == 0

    pid = s.upsert_paper(make_paper())
    assert s.revision == 1
    s.replace_chunks(pid, [Chunk(None, pid, 0, 1, "x", np.ones(2, dtype=np.float32))])
    assert s.revision == 2

    s.meta_set("library_dir", "elsewhere")
    assert s.revision == 2
    s.meta_set("embed_model", "m1")
    assert s.revision == 3
    s.meta_set("embed_model", "m1")
    assert s.revision == 3

    s.delete_paper(pid)
    assert s.revision == 4
    s.delete_paper(pid)
    assert s.revision == 4
    s.close()


def test_db_identity_is_stable_across_store_instances(tmp_path):
    db_path = tmp_path / "t.db"
    first = Store(db_path)
    identity = first.db_identity
    assert first.db_path == db_path.resolve()
    first.close()

    second = Store(db_path)
    assert second.db_identity == identity
    assert second.search_cache_key() == (identity, 0, None)
    second.close()


def test_incremental_commit_rejects_stale_revision_across_store_instances(tmp_path):
    db_path = tmp_path / "t.db"
    first = Store(db_path)
    second = Store(db_path)
    first_revision = first.index_state().revision
    stale_revision = second.index_state().revision

    first.commit_index_update(
        [
            (
                make_paper("first.pdf", title="first"),
                [Chunk(None, 0, 0, 1, "first", np.ones(2, dtype=np.float32))],
            )
        ],
        [],
        embed_model="m1",
        library_dir="root-a",
        expected_revision=first_revision,
    )

    with pytest.raises(RevisionConflictError, match="已被其他进程修改.*重试"):
        second.commit_index_update(
            [
                (
                    make_paper("second.pdf", title="second"),
                    [Chunk(None, 0, 0, 1, "second", np.ones(2, dtype=np.float32))],
                )
            ],
            [],
            embed_model="m2",
            library_dir="root-b",
            expected_revision=stale_revision,
        )

    assert first.stats() == (1, 1)
    assert first.paper_by_path("second.pdf") is None
    assert first.meta_get("embed_model") == "m1"
    assert first.meta_get("library_dir") == "root-a"
    first.close()
    second.close()


def test_replace_library_rejects_stale_revision(tmp_path):
    db_path = tmp_path / "t.db"
    first = Store(db_path)
    second = Store(db_path)
    stale_revision = second.index_state().revision
    first.upsert_paper(make_paper("kept.pdf", title="kept"))

    with pytest.raises(RevisionConflictError, match="已被其他进程修改"):
        second.replace_library(
            [(make_paper("replacement.pdf"), [])],
            embed_model="m2",
            library_dir="replacement-root",
            expected_revision=stale_revision,
        )

    assert first.stats() == (1, 0)
    assert first.paper_by_path("kept.pdf") is not None
    assert first.paper_by_path("replacement.pdf") is None
    first.close()
    second.close()


def test_list_filter(tmp_path):
    s = Store(tmp_path / "t.db")
    s.upsert_paper(make_paper("a.pdf", title="注意力机制研究", authors=["张三"]))
    s.upsert_paper(make_paper("b.pdf", title="Transformer 综述", authors=["Alice", "Bob"]))
    total, papers = s.list_papers("注意", 50, 0)
    assert total == 1 and papers[0].path == "a.pdf"
    total, papers = s.list_papers("alice", 50, 0)
    assert total == 1 and papers[0].path == "b.pdf"
    total, papers = s.list_papers(None, 1, 0)
    assert total == 2 and len(papers) == 1
    s.close()


def test_list_papers_with_chunk_counts(tmp_path):
    s = Store(tmp_path / "t.db")
    first = s.upsert_paper(make_paper("a.pdf", title="Alpha"))
    s.replace_chunks(
        first,
        [
            Chunk(None, first, 0, 1, "one", np.ones(2, dtype=np.float32)),
            Chunk(None, first, 1, 1, "two", np.ones(2, dtype=np.float32)),
        ],
    )
    s.upsert_paper(make_paper("b.pdf", title="Beta"))

    total, rows = s.list_papers_with_chunk_counts(None, 10, 0)
    assert total == 2
    assert [(paper.title, count) for paper, count in rows] == [("Alpha", 2), ("Beta", 0)]
    total, rows = s.list_papers_with_chunk_counts("Beta", 10, 0)
    assert total == 1 and rows[0][1] == 0
    s.close()


def test_cascade_delete(tmp_path):
    s = Store(tmp_path / "t.db")
    pid = s.upsert_paper(make_paper())
    s.replace_chunks(pid, [Chunk(None, pid, 0, 1, "x")])
    s.delete_paper(pid)
    assert s.get_paper(pid) is None
    assert s.get_chunks_by_paper(pid) == []
    assert s.stats() == (0, 0)
    s.close()


def test_get_chunks_join(tmp_path):
    s = Store(tmp_path / "t.db")
    pid = s.upsert_paper(make_paper("a.pdf", title="T1", authors=["张三"], year=2021))
    s.replace_chunks(pid, [Chunk(None, pid, 0, 1, "text-a"), Chunk(None, pid, 1, 2, "text-b")])
    hits = s.get_chunks([1, 2])
    assert len(hits) == 2
    h = hits[0]
    assert h.title == "T1" and h.authors == ["张三"] and h.year == 2021
    assert h.path == "a.pdf" and h.page == 1 and h.text == "text-a"
    s.close()


def test_meta_and_stats(tmp_path):
    s = Store(tmp_path / "t.db")
    s.meta_set("embed_model", "m1")
    assert s.meta_get("embed_model") == "m1"
    assert s.meta_get("nope") is None
    assert s.meta_get("schema_version") == "1"
    assert s.stats() == (0, 0)
    s.close()


def test_iter_papers(tmp_path):
    s = Store(tmp_path / "t.db")
    s.upsert_paper(make_paper("a.pdf"))
    s.upsert_paper(make_paper("b.pdf"))
    assert [p.path for p in s.iter_papers()] == ["a.pdf", "b.pdf"]
    s.close()
