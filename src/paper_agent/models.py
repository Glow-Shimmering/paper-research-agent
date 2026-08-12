"""数据模型：论文、分块、检索命中与持久化研究记录。"""
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class Paper:
    id: Optional[int]
    path: str
    sha256: str
    title: str
    authors: list[str]
    year: Optional[int]
    page_count: int
    has_text: bool
    indexed_at: str


@dataclass
class Chunk:
    id: Optional[int]
    paper_id: int
    seq: int
    page: int
    text: str
    embedding: Optional[np.ndarray] = None


@dataclass(frozen=True)
class Evidence:
    """可固定引用的论文证据快照。

    ``id`` 由论文内容哈希、分块位置和分块文本哈希确定，因此同一来源在
    重建索引、SQLite 自增 id 变化后仍得到同一个标识。``stale`` 只表示
    当前索引已无法复现这份快照；快照文本本身仍保留，便于审计旧笔记。
    """

    id: str
    paper_id: Optional[int]
    chunk_id: Optional[int]
    source_hash: str
    paper_sha256: str
    chunk_text_sha256: str
    title: str
    authors: tuple[str, ...]
    year: Optional[int]
    path: str
    page: int
    chunk_seq: int
    text: str
    annotation: str = ""
    pinned_at: Optional[str] = None
    stale: bool = False
    stale_reason: Optional[str] = None

    @property
    def evidence_id(self) -> str:
        """兼容显式 ``evidence_id`` 命名，同时保持主键字段统一为 ``id``。"""

        return self.id


@dataclass(frozen=True)
class AgentRunRecord:
    """一个可恢复的 Agent 任务运行记录。"""

    id: str
    objective: str
    status: str
    created_at: str
    updated_at: str
    plan: Any = None
    budget: Any = None
    error: Optional[str] = None

    @property
    def run_id(self) -> str:
        return self.id


@dataclass(frozen=True)
class AgentEventRecord:
    """Agent 运行中的一个有序事件。"""

    id: int
    run_id: str
    seq: int
    event_type: str
    created_at: str
    payload: Any = None

    @property
    def kind(self) -> str:
        """供偏好 ``kind`` 命名的调用方读取。"""

        return self.event_type

    @property
    def event_id(self) -> int:
        return self.id


@dataclass
class SearchHit:
    chunk_id: int
    paper_id: int
    title: str
    authors: list[str]
    year: Optional[int]
    path: str
    page: int
    text: str
    score: float


@dataclass(frozen=True)
class SearchCorpusItem:
    """检索快照中的一行；位置与 ``SearchSnapshot.embeddings`` 严格对齐。"""

    chunk_id: int
    paper_id: int
    title: str
    authors: tuple[str, ...]
    year: Optional[int]
    path: str
    page: int
    text: str


@dataclass(frozen=True)
class SearchSnapshot:
    """一次数据库读快照得到的完整检索语料。"""

    items: tuple[SearchCorpusItem, ...]
    embeddings: np.ndarray = field(repr=False, compare=False)
    embed_model: Optional[str] = None
    revision: int = 0
