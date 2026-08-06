"""测试共享辅助：造 PDF、伪嵌入器、造 Paper。"""
import hashlib
from pathlib import Path

import fitz
import numpy as np

from paper_agent.models import Paper


def make_pdf(path: Path, pages: list[str], meta: dict | None = None) -> Path:
    """用 PyMuPDF 现造一个带文本与元数据的 PDF。"""
    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        rect = fitz.Rect(50, 50, 545, 792)
        page.insert_textbox(rect, text, fontname="china-s", fontsize=11)
    if meta:
        doc.set_metadata(meta)
    doc.save(str(path))
    doc.close()
    return path


class FakeEmbedder:
    """embed(texts) 精确查表；未知文本用稳定哈希生成确定性向量。"""

    def __init__(self, model_name: str = "fake", vecs: dict[str, np.ndarray] | None = None):
        self.model_name = model_name
        self.vecs = {k: np.asarray(v, dtype=np.float32) for k, v in (vecs or {}).items()}

    @staticmethod
    def vecs_for(text: str) -> np.ndarray:
        seed = int.from_bytes(hashlib.md5(text.encode()).digest()[:8], "little")
        return np.random.default_rng(seed).random(8, dtype=np.float32)

    @property
    def dim(self) -> int:
        return 8

    def embed(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        out = []
        for t in texts:
            if t not in self.vecs:
                self.vecs[t] = self.vecs_for(t)
            out.append(self.vecs[t])
        return np.stack(out)


def make_paper(path, title="标题", year=2020, **kw):
    base = dict(
        id=None, path=path, sha256="s", title=title, authors=["A"],
        year=year, page_count=1, has_text=True, indexed_at="2026-01-01T00:00:00",
    )
    base.update(kw)
    return Paper(**base)


def noop_progress(msg: str) -> None:
    pass
