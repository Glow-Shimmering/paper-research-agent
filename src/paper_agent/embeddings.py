"""嵌入封装：fastembed 本地模型（CPU 推理，进程内缓存实例）。"""
import threading

import numpy as np

_CACHE: dict[str, tuple[object, int]] = {}
_CACHE_LOCK = threading.Lock()


class Embedder:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None
        self._dim: int | None = None

    @property
    def dim(self) -> int:
        self._load()
        assert self._dim is not None
        return self._dim

    def _load(self) -> None:
        global _CACHE
        if self._model is not None:
            return
        with _CACHE_LOCK:
            if self._model is not None:
                return
            if self.model_name in _CACHE:
                self._model, self._dim = _CACHE[self.model_name]
                return
            try:
                from fastembed import TextEmbedding

                model = TextEmbedding(model_name=self.model_name)
            except Exception as exc:  # 网络、模型不存在等
                raise RuntimeError(
                    f"嵌入模型加载失败：{exc}\n"
                    "检查网络，或设置 HF_ENDPOINT=https://hf-mirror.com 后重试"
                ) from exc
            dim = model.embedding_size
            _CACHE[self.model_name] = (model, dim)
            self._model, self._dim = model, dim

    def embed(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """返回 (N, D) float32 矩阵。"""
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        self._load()
        vectors = list(self._model.embed(texts, batch_size=batch_size))  # type: ignore[union-attr]
        return np.stack(vectors)
