"""BM25 关键词索引：jieba 分词 + rank_bm25。"""
import jieba
from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    """分词规则：jieba.cut_for_search + 小写；保留含 CJK 的 token 或长度 ≥2 的字母数字 token。"""
    tokens: list[str] = []
    for tok in jieba.cut_for_search(text.lower()):
        tok = tok.strip()
        if not tok:
            continue
        if any("\u4e00" <= ch <= "\u9fff" for ch in tok):
            tokens.append(tok)
        elif tok.isalnum() and len(tok) >= 2:
            tokens.append(tok)
    return tokens


class Bm25Index:
    def __init__(self) -> None:
        self._bm25: BM25Okapi | None = None

    def build(self, chunks: list[str]) -> None:
        self._bm25 = BM25Okapi([tokenize(c) for c in chunks])

    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        """返回 (语料下标, 得分) 降序列表，仅含得分 > 0 的命中。"""
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [(i, float(scores[i])) for i in order if scores[i] > 0][:top_k]
