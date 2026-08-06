"""数据模型：论文、分块、检索命中。"""
from dataclasses import dataclass, field
from typing import Optional

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
