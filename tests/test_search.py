import hashlib

import numpy as np
import pytest

from paper_agent.models import Chunk, Paper
from paper_agent.search import hybrid_search, rrf_fuse
from paper_agent.store import Store


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
