import hashlib
import threading

import numpy as np
import pytest

import pragent.search as search_module
from pragent.models import Chunk, Paper
from pragent.search import hybrid_search, rrf_fuse, search_within_paper
from pragent.store import Store


class FakeEmbedder:
    """embed(texts) 精确查表；未知文本用稳定哈希生成确定性向量。"""

    def __init__(self, vecs: dict[str, np.ndarray] | None = None):
        self.vecs = {k: np.asarray(v, dtype=np.float32) for k, v in (vecs or {}).items()}
        self.model_name = "fake"

    def embed(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        out = []
        for t in texts:
            if t not in self.vecs:
                seed = int.from_bytes(hashlib.md5(t.encode()).digest()[:8], "little")
                self.vecs[t] = np.random.default_rng(seed).random(8, dtype=np.float32)
            out.append(self.vecs[t])
        return np.stack(out)


def make_paper(path, title="标题", year=2020, **kw):
    base = dict(
        id=None, path=path, sha256="s", title=title, authors=["A"],
        year=year, page_count=1, has_text=True, indexed_at="2026-01-01T00:00:00",
    )
    base.update(kw)
    return Paper(**base)


def seed_store(tmp_path, texts_per_paper: dict[str, list[str]]) -> tuple[Store, FakeEmbedder]:
    """texts_per_paper: {path: [chunk_text, ...]}；每块给一个 8 维单位基向量。"""
    s = Store(tmp_path / "t.db")
    vecs: dict[str, np.ndarray] = {}
    for path, texts in texts_per_paper.items():
        pid = s.upsert_paper(make_paper(path))
        chunks = []
        for i, t in enumerate(texts):
            v = np.zeros(8, dtype=np.float32)
            v[i % 8] = 1.0
            vecs[t] = v
            chunks.append(Chunk(None, pid, i, 1, t, v))
        s.replace_chunks(pid, chunks)
    return s, FakeEmbedder(vecs)


def test_empty_library(tmp_path):
    s = Store(tmp_path / "t.db")
    hits = hybrid_search(s, FakeEmbedder(), "随便查查")
    assert hits == []


def test_embedding_model_mismatch_fails_before_query_embedding(tmp_path):
    s, emb = seed_store(tmp_path, {"a.pdf": ["alpha"]})
    s.meta_set("embed_model", "indexed-model")
    emb.model_name = "query-model"

    with pytest.raises(RuntimeError, match="索引由嵌入模型"):
        hybrid_search(s, emb, "alpha")
    s.close()


def test_search_remains_on_one_snapshot_during_reindex(tmp_path):
    s = Store(tmp_path / "t.db")
    old_vector = np.zeros(8, dtype=np.float32)
    old_vector[0] = 1.0
    pid = s.upsert_paper(
        make_paper("a.pdf", title="旧标题"),
        [Chunk(None, 0, 0, 1, "alpha 旧分块", old_vector)],
    )
    s.meta_set("embed_model", "fake")

    started = threading.Event()
    resume = threading.Event()

    class BlockingEmbedder(FakeEmbedder):
        def embed(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
            started.set()
            if not resume.wait(timeout=2):
                raise AssertionError("等待并发重索引超时")
            return super().embed(texts, batch_size)

    emb = BlockingEmbedder({"alpha": old_vector})
    result: list[list] = []
    errors: list[BaseException] = []

    def run_search():
        try:
            result.append(hybrid_search(s, emb, "alpha", top=1))
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=run_search)
    thread.start()
    try:
        assert started.wait(timeout=2)
        new_vector = np.zeros(8, dtype=np.float32)
        new_vector[1] = 1.0
        s.upsert_paper(
            make_paper("a.pdf", title="新标题"),
            [Chunk(None, pid, 0, 1, "完全不同的新分块", new_vector)],
        )
    finally:
        resume.set()
        thread.join(timeout=2)

    assert not thread.is_alive()
    assert errors == []
    assert result and result[0][0].text == "alpha 旧分块"
    assert result[0][0].title == "旧标题"
    s.close()


def test_corpus_cache_reuses_bm25_at_same_revision(tmp_path, monkeypatch):
    s, emb = seed_store(tmp_path, {"a.pdf": ["alpha 第一块", "beta 第二块"]})
    build_calls = 0
    original_build = search_module.Bm25Index.build

    def counting_build(self, chunks):
        nonlocal build_calls
        build_calls += 1
        return original_build(self, chunks)

    monkeypatch.setattr(search_module.Bm25Index, "build", counting_build)
    hybrid_search(s, emb, "alpha")
    hybrid_search(s, emb, "beta")

    assert build_calls == 1
    s.close()


def test_corpus_cache_refreshes_when_revision_changes(tmp_path, monkeypatch):
    s, emb = seed_store(tmp_path, {"a.pdf": ["alpha 旧分块"]})
    build_calls = 0
    original_build = search_module.Bm25Index.build

    def counting_build(self, chunks):
        nonlocal build_calls
        build_calls += 1
        return original_build(self, chunks)

    monkeypatch.setattr(search_module.Bm25Index, "build", counting_build)
    first = hybrid_search(s, emb, "alpha", top=1)

    new_vector = np.zeros(8, dtype=np.float32)
    new_vector[0] = 1.0
    s.replace_chunks(1, [Chunk(None, 1, 0, 1, "alpha 新分块", new_vector)])
    emb.vecs["alpha"] = new_vector
    second = hybrid_search(s, emb, "alpha", top=1)

    assert build_calls == 2
    assert sum(key[0] == s.db_identity for key in search_module._CORPUS_CACHE) == 1
    assert first[0].text == "alpha 旧分块"
    assert second[0].text == "alpha 新分块"
    s.close()


def test_corpus_cache_build_is_single_flight_per_revision(tmp_path, monkeypatch):
    s, emb = seed_store(tmp_path, {"a.pdf": ["alpha"]})
    original_snapshot = s.search_snapshot
    original_build = search_module._build_cached_corpus
    snapshot_count = 0
    build_calls = 0
    counter_lock = threading.Lock()
    both_snapshots = threading.Event()

    def counting_snapshot():
        nonlocal snapshot_count
        snapshot = original_snapshot()
        with counter_lock:
            snapshot_count += 1
            if snapshot_count == 2:
                both_snapshots.set()
        return snapshot

    def blocking_build(snapshot):
        nonlocal build_calls
        with counter_lock:
            build_calls += 1
        assert both_snapshots.wait(timeout=2)
        return original_build(snapshot)

    monkeypatch.setattr(s, "search_snapshot", counting_snapshot)
    monkeypatch.setattr(search_module, "_build_cached_corpus", blocking_build)
    barrier = threading.Barrier(3)
    results = []
    errors = []

    def run_search():
        try:
            barrier.wait(timeout=2)
            results.append(hybrid_search(s, emb, "alpha"))
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=run_search) for _ in range(2)]
    for thread in threads:
        thread.start()
    barrier.wait(timeout=2)
    for thread in threads:
        thread.join(timeout=3)

    assert all(not thread.is_alive() for thread in threads)
    assert errors == []
    assert len(results) == 2
    assert build_calls == 1
    s.close()


def test_bm25_exact_word_hit(tmp_path):
    s, emb = seed_store(
        tmp_path,
        {
            "a.pdf": ["transformer 架构的编码器设计", "卷积网络在图像上的应用"],
            "b.pdf": ["注意力机制综述"],
        },
    )
    hits = hybrid_search(s, emb, "transformer", top=10)
    assert hits and hits[0].text.startswith("transformer")
    assert hits[0].title == "标题"


def test_cosine_semantic_hit(tmp_path):
    # 查询向量 = e0；仅"zzzz语义块"的向量为 e0 → 余弦命中第一
    s, emb = seed_store(
        tmp_path,
        {"a.pdf": ["aaaaa", "bbbbb", "zzzz语义块"]},
    )
    pid = 1
    e0 = np.zeros(8, dtype=np.float32); e0[0] = 1.0
    e3 = np.zeros(8, dtype=np.float32); e3[3] = 1.0
    e5 = np.zeros(8, dtype=np.float32); e5[5] = 1.0
    s.replace_chunks(
        pid,
        [
            Chunk(None, pid, 0, 1, "aaaaa", e3),
            Chunk(None, pid, 1, 1, "bbbbb", e5),
            Chunk(None, pid, 2, 1, "zzzz语义块", e0),
        ],
    )
    emb.vecs["zzzz查询词"] = e0
    hits = hybrid_search(s, emb, "zzzz查询词", top=3)
    assert hits and hits[0].text == "zzzz语义块"


def test_rrf_fusion_order(tmp_path):
    # BM25 仅命中 B（唯一含 alpha 的块）；向量余弦 A(1.0) > B(0.9) > C(0.8)
    # 融合分：B = 1/61 + 1/62 > A = 1/61 > C = 1/63 → 顺序 B, A, C
    s = Store(tmp_path / "t.db")
    pid = s.upsert_paper(make_paper("a.pdf"))
    e0 = np.zeros(8, dtype=np.float32); e0[0] = 1.0
    vb = 0.9 * e0 + 0.1 * np.roll(e0, 1)
    vc = 0.8 * e0 + 0.2 * np.roll(e0, 2)
    s.replace_chunks(
        pid,
        [
            Chunk(None, pid, 0, 1, "A块内容", e0),
            Chunk(None, pid, 1, 1, "B块 alpha", vb),
            Chunk(None, pid, 2, 1, "C块内容", vc),
        ],
    )
    hits = hybrid_search(s, FakeEmbedder({"alpha": e0}), "alpha", top=3)
    assert [h.text for h in hits] == ["B块 alpha", "A块内容", "C块内容"]


def test_per_paper_cap(tmp_path):
    s, emb = seed_store(
        tmp_path,
        {
            "a.pdf": ["alpha 内容一", "alpha 内容二", "alpha 内容三"],
            "b.pdf": ["alpha 别的论文"],
        },
    )
    hits = hybrid_search(s, emb, "alpha", top=10, per_paper_cap=1)
    papers = [h.paper_id for h in hits]
    assert len(papers) == 2
    assert len(set(papers)) == 2  # 每篇最多一条


def test_rrf_fuse_math():
    fused = rrf_fuse([(0, 1.0), (1, 0.5)], [(1, 0.9), (0, 0.8)])
    assert fused[0] == pytest.approx(1 / 61 + 1 / 62)
    assert fused[1] == pytest.approx(1 / 62 + 1 / 61)


def test_search_within_paper_requests_embeddings_and_accepts_dict_rows():
    e0 = np.zeros(8, dtype=np.float32)
    e0[0] = 1.0
    e1 = np.zeros(8, dtype=np.float32)
    e1[1] = 1.0

    class RecordingStore:
        def __init__(self):
            self.include_embeddings_calls = []

        def paper_by_id(self, paper_id):
            return {
                "id": paper_id,
                "title": "字典论文",
                "authors": ["A"],
                "year": 2025,
                "path": "dict.pdf",
            }

        def paper_chunks(self, paper_id, include_embeddings=False):
            self.include_embeddings_calls.append(include_embeddings)
            return [
                {
                    "id": 11,
                    "paper_id": paper_id,
                    "seq": 0,
                    "page": 1,
                    "text": "完全无关",
                    "embedding": e1 if include_embeddings else None,
                },
                {
                    "id": 12,
                    "paper_id": paper_id,
                    "seq": 1,
                    "page": 2,
                    "text": "语义目标",
                    "embedding": e0 if include_embeddings else None,
                },
            ]

        def meta_get(self, key):
            return "fake" if key == "embed_model" else None

    store = RecordingStore()
    hits = search_within_paper(
        store,
        FakeEmbedder({"semantic query": e0}),
        7,
        "semantic query",
        top=2,
    )

    assert store.include_embeddings_calls == [True]
    assert hits and hits[0].chunk_id == 12
    assert hits[0].paper_id == 7
    assert hits[0].title == "字典论文"


def test_search_within_paper_rejects_embedding_model_mismatch(tmp_path):
    s, emb = seed_store(tmp_path, {"a.pdf": ["alpha"]})
    s.meta_set("embed_model", "indexed-model")
    emb.model_name = "query-model"

    with pytest.raises(RuntimeError, match="索引由嵌入模型"):
        search_within_paper(s, emb, 1, "alpha")
    s.close()
