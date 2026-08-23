"""数据模型：论文、分块、检索命中与持久化研究记录。"""
from dataclasses import dataclass, field
from typing import Any, Generic, Optional, TypeVar

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


T = TypeVar("T")


@dataclass(frozen=True)
class Page(Generic[T]):
    """Repository 的稳定分页返回值。"""

    total: int
    items: tuple[T, ...]
    limit: int
    offset: int


@dataclass(frozen=True)
class ResearchProject:
    id: str
    title: str
    description: str
    default_language: str
    citation_style: str
    status: str
    version: int
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class ResearchQuestion:
    id: str
    project_id: str
    question: str
    position: int
    version: int
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class ResearchSource:
    id: str
    canonical_key: str
    source_kind: str
    title: str
    authors: tuple[str, ...]
    year: Optional[int]
    doi: Optional[str]
    arxiv_id: Optional[str]
    canonical_url: Optional[str]
    content_sha256: Optional[str]
    indexed_paper_id: Optional[int]
    status: str
    metadata: Any
    locator: Any
    snapshot_path: Optional[str]
    snapshot_sha256: Optional[str]
    extracted_text: Optional[str] = field(repr=False, compare=False)
    fetched_at: Optional[str]
    version: int
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class SourceIdentity:
    id: str
    source_id: str
    identity_kind: str
    normalized_value: str
    is_primary: bool
    created_at: str


@dataclass(frozen=True)
class SourceRecord:
    id: str
    source_id: str
    provider: str
    provider_record_id: str
    record_url: Optional[str]
    raw_metadata: Any
    retrieved_at: str
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class ProjectSourceMembership:
    project_id: str
    source: ResearchSource
    position: int
    note: str
    added_at: str


@dataclass(frozen=True)
class ResearchArtifact:
    id: str
    project_id: str
    source_id: Optional[str]
    artifact_type: str
    title: str
    status: str
    current_revision_number: int
    version: int
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class ArtifactRevision:
    id: str
    artifact_id: str
    revision_number: int
    parent_revision_id: Optional[str]
    content: Any
    created_by: str
    source_fingerprint: Optional[str]
    model: Optional[str]
    usage: Any
    finish_reason: Optional[str]
    prompt_version: Optional[str]
    schema_version: Optional[int]
    created_at: str


@dataclass(frozen=True)
class ArtifactEvidenceLink:
    artifact_revision_id: str
    evidence_id: str
    field_path: str
    ordinal: int
    created_at: str


@dataclass(frozen=True)
class ArtifactFreshness:
    artifact_id: str
    revision_id: Optional[str]
    stale: bool
    saved_fingerprint: Optional[str]
    current_fingerprint: Optional[str]
    reason: Optional[str] = None


@dataclass(frozen=True)
class ResearchNote:
    id: str
    project_id: str
    scope_kind: str
    source_id: Optional[str]
    evidence_id: Optional[str]
    title: str
    content_markdown: str
    version: int
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class ResearchJob:
    id: str
    project_id: Optional[str]
    artifact_id: Optional[str]
    job_type: str
    status: str
    payload: Any
    result: Any
    error_code: Optional[str]
    error_message: Optional[str]
    progress_current: int
    progress_total: Optional[int]
    attempts: int
    max_attempts: int
    idempotent: bool
    priority: int
    run_after: Optional[str]
    timeout_seconds: Optional[int]
    idempotency_key: Optional[str]
    lease_owner: Optional[str]
    lease_expires_at: Optional[str]
    cancel_requested_at: Optional[str]
    started_at: Optional[str]
    finished_at: Optional[str]
    version: int
    created_at: str
    updated_at: str
