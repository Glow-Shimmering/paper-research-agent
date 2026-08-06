"""混合检索：BM25 + 向量余弦，RRF 融合。"""
import numpy as np

from .bm25 import Bm25Index
from .models import SearchHit

_RRF_K = 60
_RAW_TOP = 100  # 融合前各自取前 100 名


def rrf_fuse(
    bm25_hits: list[tuple[int, float]],
    vec_hits: list[tuple[int, float]],
    k: int = _RRF_K,
) -> dict[int, float]:
    fused: dict[int, float] = {}
    for rank, (idx, _) in enumerate(bm25_hits, start=1):
        fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank)
    for rank, (idx, _) in enumerate(vec_hits, start=1):
        fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank)
    return fused


def hybrid_search(store, embedder, query: str, top: int = 10, per_paper_cap: int | None = None) -> list[SearchHit]:
    """混合检索。空库返回 []。per_paper_cap 非空时每篇论文最多保留 cap 条。"""
    chunks = store.all_chunks()
    if not chunks:
        return []
    corpus = [c.text for c in chunks]
    ids = [c.id for c in chunks]
    paper_ids = [c.paper_id for c in chunks]

    bm25 = Bm25Index()
    bm25.build(corpus)
    bm25_hits = bm25.search(query, top_k=_RAW_TOP)

    matrix, _ = store.all_embeddings()  # 与 all_chunks 同序：按 id 升序、仅含非空嵌入
    q_vec = embedder.embed([query])[0]
    norms = np.linalg.norm(matrix, axis=1)
    norms[norms == 0] = 1.0
    q_norm = q_vec / (np.linalg.norm(q_vec) or 1.0)
    cos = (matrix / norms[:, None]) @ q_norm
    vec_hits = [(int(i), float(cos[i])) for i in np.argsort(-cos) if cos[i] > 0][:_RAW_TOP]

    ranked = sorted(rrf_fuse(bm25_hits, vec_hits).items(), key=lambda kv: kv[1], reverse=True)

    if per_paper_cap is not None:
        seen: dict[int, int] = {}
        kept: list[tuple[int, float]] = []
        for idx, score in ranked:
            pid = paper_ids[idx]
            if seen.get(pid, 0) >= per_paper_cap:
                continue
            seen[pid] = seen.get(pid, 0) + 1
            kept.append((idx, score))
        ranked = kept

    ranked = ranked[:top]
    if not ranked:
        return []
    hits = store.get_chunks([ids[idx] for idx, _ in ranked])
    for hit, (_, score) in zip(hits, ranked):
        hit.score = score
    return hits
