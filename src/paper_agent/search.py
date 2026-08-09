"""混合检索：BM25 + 向量余弦，RRF 融合。"""
from collections import OrderedDict
from dataclasses import dataclass, field
import threading

import numpy as np

from .bm25 import Bm25Index
from .models import SearchCorpusItem, SearchHit, SearchSnapshot

_RRF_K = 60
_RAW_TOP = 100  # 融合前各自取前 100 名
_CACHE_MAX_ENTRIES = 8


@dataclass(frozen=True)
class _CachedCorpus:
    items: tuple[SearchCorpusItem, ...]
    bm25: Bm25Index | None = field(repr=False, compare=False)
    normalized_embeddings: np.ndarray = field(repr=False, compare=False)
    embed_model: str | None = None
    revision: int = 0


_CORPUS_CACHE: OrderedDict[tuple[str, int, str | None], _CachedCorpus] = OrderedDict()
_CORPUS_CACHE_LOCK = threading.RLock()
_CORPUS_BUILD_LOCKS: dict[tuple[str, int, str | None], threading.Lock] = {}


def _cache_get(key: tuple[str, int, str | None]) -> _CachedCorpus | None:
    corpus = _CORPUS_CACHE.get(key)
    if corpus is not None:
        _CORPUS_CACHE.move_to_end(key)
    return corpus


def _build_cached_corpus(snapshot: SearchSnapshot) -> _CachedCorpus:
    if not snapshot.items:
        normalized = np.zeros((0, 0), dtype=np.float32)
        bm25 = None
    else:
        bm25 = Bm25Index()
        bm25.build([item.text for item in snapshot.items])
        norms = np.linalg.norm(snapshot.embeddings, axis=1)
        norms[norms == 0] = 1.0
        normalized = snapshot.embeddings / norms[:, None]
    normalized.setflags(write=False)
    return _CachedCorpus(
        items=snapshot.items,
        bm25=bm25,
        normalized_embeddings=normalized,
        embed_model=snapshot.embed_model,
        revision=snapshot.revision,
    )


def _get_cached_corpus(store) -> _CachedCorpus:
    requested_key = store.search_cache_key()
    with _CORPUS_CACHE_LOCK:
        cached = _cache_get(requested_key)
        if cached is not None:
            return cached

    snapshot = store.search_snapshot()
    actual_key = (store.db_identity, snapshot.revision, snapshot.embed_model)
    with _CORPUS_CACHE_LOCK:
        cached = _cache_get(actual_key)
        if cached is not None:
            return cached
        build_lock = _CORPUS_BUILD_LOCKS.setdefault(actual_key, threading.Lock())

    # 构建可能需要对整库分词和归一化；仅锁住当前版本，不阻塞其他
    # 数据库的 cache hit/miss。并发 miss 由 per-key single-flight 合并。
    with build_lock:
        with _CORPUS_CACHE_LOCK:
            cached = _cache_get(actual_key)
            if cached is not None:
                return cached
        try:
            candidate = _build_cached_corpus(snapshot)
        except Exception:
            with _CORPUS_CACHE_LOCK:
                if _CORPUS_BUILD_LOCKS.get(actual_key) is build_lock:
                    _CORPUS_BUILD_LOCKS.pop(actual_key, None)
            raise
        with _CORPUS_CACHE_LOCK:
            cached = _cache_get(actual_key)
            if cached is None:
                # 同一数据库只保留最新版本，避免频繁增量索引时在内存中
                # 滞留多份完整语料；全局上限服务多个独立数据库。
                for key in list(_CORPUS_CACHE):
                    if key[0] == actual_key[0] and key != actual_key:
                        del _CORPUS_CACHE[key]
                _CORPUS_CACHE[actual_key] = candidate
                while len(_CORPUS_CACHE) > _CACHE_MAX_ENTRIES:
                    _CORPUS_CACHE.popitem(last=False)
                cached = candidate
            _CORPUS_BUILD_LOCKS.pop(actual_key, None)
            return cached


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
    corpus_cache = _get_cached_corpus(store)
    items = corpus_cache.items
    if not items:
        return []
    current_model = getattr(embedder, "model_name", None)
    if corpus_cache.embed_model is not None and corpus_cache.embed_model != current_model:
        raise RuntimeError(
            f"索引由嵌入模型「{corpus_cache.embed_model}」建立，"
            f"当前查询模型为「{current_model}」。请切换回原模型，或使用 --force 全量重建索引。"
        )

    paper_ids = [item.paper_id for item in items]

    assert corpus_cache.bm25 is not None
    bm25_hits = corpus_cache.bm25.search(query, top_k=_RAW_TOP)

    matrix = corpus_cache.normalized_embeddings
    q_vec = np.asarray(embedder.embed([query])[0], dtype=np.float32)
    if q_vec.ndim != 1 or q_vec.shape[0] != matrix.shape[1]:
        actual = q_vec.shape[0] if q_vec.ndim == 1 else tuple(q_vec.shape)
        raise RuntimeError(
            f"查询嵌入维度（{actual}）与索引维度（{matrix.shape[1]}）不一致，"
            "请确认嵌入模型配置，或使用 --force 全量重建索引。"
        )
    q_norm = q_vec / (np.linalg.norm(q_vec) or 1.0)
    cos = matrix @ q_norm
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
    return [
        SearchHit(
            chunk_id=items[idx].chunk_id,
            paper_id=items[idx].paper_id,
            title=items[idx].title,
            authors=list(items[idx].authors),
            year=items[idx].year,
            path=items[idx].path,
            page=items[idx].page,
            text=items[idx].text,
            score=score,
        )
        for idx, score in ranked
    ]
