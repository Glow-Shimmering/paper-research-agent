"""SQLite 存储层：索引、证据与 Agent 运行记录。"""
import hashlib
import json
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np

from .models import (
    AgentEventRecord,
    AgentRunRecord,
    Chunk,
    Evidence,
    Paper,
    SearchCorpusItem,
    SearchHit,
    SearchSnapshot,
)
from .storage.migrations import MigrationReport, migrate_schema

_AGENT_RUN_STATUSES = frozenset(
    {
        "proposed",
        "running",
        "awaiting_confirmation",
        "succeeded",
        "failed",
        "cancelled",
        "blocked",
    }
)


class RevisionConflictError(RuntimeError):
    """索引预处理期间数据库已变化；本批写入必须放弃。"""


class AgentRunStatusConflictError(RuntimeError):
    """Agent Run 的 compare-and-swap 状态转换失败。"""


@dataclass(frozen=True)
class IndexState:
    """索引任务开始时的一致数据库快照（不包含大体积嵌入）。"""

    revision: int
    embed_model: Optional[str]
    library_dir: Optional[str]
    papers: tuple[Paper, ...]
    paper_ids_with_chunks: frozenset[int]
    has_search_corpus: bool


class Store:
    def __init__(self, db_path: str | Path):
        db_value = str(db_path)
        self._db_path = (
            None if db_value == ":memory:" else Path(db_value).expanduser().resolve(strict=False)
        )
        self._conn = sqlite3.connect(db_value, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        try:
            with self._lock:
                self._conn.execute("PRAGMA foreign_keys=ON")
                self._conn.execute("PRAGMA busy_timeout=5000")
                self._migration_report = migrate_schema(
                    self._conn,
                    db_path=self._db_path,
                )
                self._conn.execute("PRAGMA journal_mode=WAL")
                self._conn.execute(
                    "INSERT OR IGNORE INTO meta (key, value) VALUES ('index_revision', '0')"
                )
                self._conn.execute(
                    "INSERT OR IGNORE INTO meta (key, value) VALUES ('db_identity', ?)",
                    (str(uuid.uuid4()),),
                )
                identity_row = self._conn.execute(
                    "SELECT value FROM meta WHERE key='db_identity'"
                ).fetchone()
                self._conn.commit()
        except Exception:
            self._conn.close()
            raise
        location = str(self._db_path) if self._db_path is not None else ":memory:"
        self._db_identity = f"{location}|{identity_row['value']}"

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @property
    def db_path(self) -> Optional[Path]:
        """数据库规范化路径；内存数据库返回 ``None``。"""
        return self._db_path

    @property
    def db_identity(self) -> str:
        """跨 Store 实例稳定、跨数据库文件隔离的缓存标识。"""
        return self._db_identity

    @property
    def migration_report(self) -> MigrationReport:
        """本次打开数据库时的 schema migration 结果。"""
        return self._migration_report

    # ---------- papers ----------

    def upsert_paper(self, paper: Paper, chunks: Optional[list[Chunk]] = None) -> int:
        """按 path 幂等写入；chunks 非空时同事务替换该论文所有分块。"""
        with self._lock, self._conn:
            paper_id = self._upsert_paper_locked(paper, chunks)
            self._bump_revision_locked()
        return paper_id

    def replace_library(
        self,
        entries: list[tuple[Paper, list[Chunk]]],
        *,
        embed_model: str,
        library_dir: str,
        expected_revision: int,
    ) -> None:
        """核对版本后，在一个跨连接互斥事务中原子替换整库。"""
        with self._lock:
            self._begin_immediate_locked()
            try:
                self._assert_revision_locked(expected_revision)
                self._conn.execute("DELETE FROM papers")
                for paper, chunks in entries:
                    self._upsert_paper_locked(paper, chunks)
                self._set_meta_locked("embed_model", embed_model)
                self._set_meta_locked("library_dir", library_dir)
                self._bump_revision_locked()
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def commit_index_update(
        self,
        entries: list[tuple[Paper, list[Chunk]]],
        delete_paper_ids: list[int],
        *,
        embed_model: str,
        library_dir: Optional[str],
        expected_revision: int,
    ) -> None:
        """原子提交一批增量索引变更；版本冲突时不写入任何内容。"""
        with self._lock:
            self._begin_immediate_locked()
            try:
                self._assert_revision_locked(expected_revision)
                changed = False
                for paper, chunks in entries:
                    self._upsert_paper_locked(paper, chunks)
                    changed = True
                for paper_id in delete_paper_ids:
                    cursor = self._conn.execute("DELETE FROM papers WHERE id=?", (paper_id,))
                    changed = bool(cursor.rowcount) or changed
                changed = self._set_meta_locked("embed_model", embed_model) or changed
                if library_dir is not None:
                    changed = self._set_meta_locked("library_dir", library_dir) or changed
                if changed:
                    self._bump_revision_locked()
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def index_state(self) -> IndexState:
        """返回索引预处理所需的小型一致快照及 expected revision。"""
        with self._lock:
            self._conn.execute("BEGIN")
            try:
                meta_rows = self._conn.execute(
                    """SELECT key, value FROM meta
                       WHERE key IN ('index_revision', 'embed_model', 'library_dir')"""
                ).fetchall()
                paper_rows = self._conn.execute("SELECT * FROM papers ORDER BY id").fetchall()
                chunk_rows = self._conn.execute(
                    "SELECT DISTINCT paper_id FROM chunks"
                ).fetchall()
                corpus_row = self._conn.execute(
                    "SELECT EXISTS(SELECT 1 FROM chunks WHERE embedding IS NOT NULL) AS present"
                ).fetchone()
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

        meta = {row["key"]: row["value"] for row in meta_rows}
        return IndexState(
            revision=self._parse_revision(meta.get("index_revision")),
            embed_model=meta.get("embed_model"),
            library_dir=meta.get("library_dir"),
            papers=tuple(self._paper_from_row(row) for row in paper_rows),
            paper_ids_with_chunks=frozenset(int(row["paper_id"]) for row in chunk_rows),
            has_search_corpus=bool(corpus_row["present"]),
        )

    def delete_paper(self, paper_id: int) -> None:
        with self._lock, self._conn:
            cursor = self._conn.execute("DELETE FROM papers WHERE id=?", (paper_id,))
            if cursor.rowcount:
                self._bump_revision_locked()

    def get_paper(self, paper_id: int) -> Optional[Paper]:
        with self._lock:
            row = self._conn.execute("SELECT * FROM papers WHERE id=?", (paper_id,)).fetchone()
        return self._paper_from_row(row) if row else None

    def paper_by_id(self, paper_id: int) -> Optional[Paper]:
        """按主键取得论文；与 ``paper_by_path`` 组成文档导航接口。"""

        return self.get_paper(paper_id)

    def paper_by_path(self, path: str) -> Optional[Paper]:
        with self._lock:
            row = self._conn.execute("SELECT * FROM papers WHERE path=?", (path,)).fetchone()
        return self._paper_from_row(row) if row else None

    def list_papers(self, q: Optional[str], limit: int, offset: int) -> tuple[int, list[Paper]]:
        with self._lock:
            if q:
                like = f"%{q}%"
                total = self._conn.execute(
                    "SELECT COUNT(*) AS n FROM papers WHERE title LIKE ? OR authors LIKE ?",
                    (like, like),
                ).fetchone()["n"]
                rows = self._conn.execute(
                    "SELECT * FROM papers WHERE title LIKE ? OR authors LIKE ? ORDER BY id LIMIT ? OFFSET ?",
                    (like, like, limit, offset),
                ).fetchall()
            else:
                total = self._conn.execute("SELECT COUNT(*) AS n FROM papers").fetchone()["n"]
                rows = self._conn.execute(
                    "SELECT * FROM papers ORDER BY id LIMIT ? OFFSET ?", (limit, offset)
                ).fetchall()
        return total, [self._paper_from_row(r) for r in rows]

    def list_papers_with_chunk_counts(
        self, q: Optional[str], limit: int, offset: int
    ) -> tuple[int, list[tuple[Paper, int]]]:
        """分页列论文并在同一聚合查询返回分块数，避免 Web N+1 查询。"""
        with self._lock:
            params: list[object] = []
            where = ""
            if q:
                where = "WHERE p.title LIKE ? OR p.authors LIKE ?"
                like = f"%{q}%"
                params.extend([like, like])
            total = self._conn.execute(
                f"SELECT COUNT(*) AS n FROM papers p {where}", params
            ).fetchone()["n"]
            rows = self._conn.execute(
                f"""SELECT p.*, COUNT(c.id) AS chunk_count
                    FROM papers p LEFT JOIN chunks c ON c.paper_id = p.id
                    {where}
                    GROUP BY p.id
                    ORDER BY p.id LIMIT ? OFFSET ?""",
                [*params, limit, offset],
            ).fetchall()
        return total, [
            (self._paper_from_row(row), int(row["chunk_count"])) for row in rows
        ]

    def iter_papers(self) -> Iterator[Paper]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM papers ORDER BY id").fetchall()
        return iter([self._paper_from_row(r) for r in rows])

    # ---------- chunks ----------

    def replace_chunks(self, paper_id: int, chunks: list[Chunk]) -> None:
        with self._lock, self._conn:
            self._replace_chunks_locked(paper_id, chunks)
            self._bump_revision_locked()

    def _replace_chunks_locked(self, paper_id: int, chunks: list[Chunk]) -> None:
        self._conn.execute("DELETE FROM chunks WHERE paper_id=?", (paper_id,))
        for c in chunks:
            blob = c.embedding.astype(np.float32).tobytes() if c.embedding is not None else None
            self._conn.execute(
                "INSERT INTO chunks (paper_id, seq, page, text, embedding) VALUES (?, ?, ?, ?, ?)",
                (paper_id, c.seq, c.page, c.text, blob),
            )

    def _upsert_paper_locked(self, paper: Paper, chunks: Optional[list[Chunk]]) -> int:
        self._conn.execute(
            """INSERT INTO papers (path, sha256, title, authors, year, page_count, has_text, indexed_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(path) DO UPDATE SET
                   sha256=excluded.sha256, title=excluded.title, authors=excluded.authors,
                   year=excluded.year, page_count=excluded.page_count,
                   has_text=excluded.has_text, indexed_at=excluded.indexed_at""",
            (
                paper.path,
                paper.sha256,
                paper.title,
                json.dumps(paper.authors, ensure_ascii=False),
                paper.year,
                paper.page_count,
                int(paper.has_text),
                paper.indexed_at,
            ),
        )
        # DO UPDATE 分支不更新连接级 last_insert_rowid，必须按 path 回查。
        row = self._conn.execute("SELECT id FROM papers WHERE path=?", (paper.path,)).fetchone()
        paper_id = row["id"]
        if chunks is not None:
            self._replace_chunks_locked(paper_id, chunks)
        return paper_id

    def get_chunks_by_paper(self, paper_id: int, include_embeddings: bool = False) -> list[Chunk]:
        cols = "id, paper_id, seq, page, text" + (", embedding" if include_embeddings else "")
        with self._lock:
            rows = self._conn.execute(
                f"SELECT {cols} FROM chunks WHERE paper_id=? ORDER BY seq", (paper_id,)
            ).fetchall()
        return [self._chunk_from_row(row, include_embedding=include_embeddings) for row in rows]

    def paper_chunks(self, paper_id: int, include_embeddings: bool = False) -> list[Chunk]:
        """返回论文的全部分块，按文档顺序排列。"""

        return self.get_chunks_by_paper(paper_id, include_embeddings=include_embeddings)

    def chunk_context(
        self,
        chunk_id: int,
        before: int = 1,
        after: int = 1,
        *,
        radius: Optional[int] = None,
    ) -> list[Chunk]:
        """返回目标分块及其前后文；目标不存在时返回空列表。"""

        if radius is not None:
            before = after = radius
        if before < 0 or after < 0:
            raise ValueError("before、after 和 radius 必须是非负整数")
        with self._lock:
            rows = self._conn.execute(
                """SELECT context.id, context.paper_id, context.seq,
                          context.page, context.text
                   FROM chunks AS target
                   JOIN chunks AS context
                     ON context.paper_id = target.paper_id
                    AND context.seq BETWEEN target.seq - ? AND target.seq + ?
                   WHERE target.id=?
                   ORDER BY context.seq""",
                (before, after, chunk_id),
            ).fetchall()
        return [self._chunk_from_row(row) for row in rows]

    def all_embeddings(self) -> tuple[np.ndarray, list[int]]:
        """返回 (N×D float32 矩阵, chunk_id 列表)，按 chunk id 升序。"""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, embedding FROM chunks WHERE embedding IS NOT NULL ORDER BY id"
            ).fetchall()
        if not rows:
            return np.zeros((0, 0), dtype=np.float32), []
        ids = [r["id"] for r in rows]
        matrix = np.vstack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
        return matrix, ids

    def all_chunks(self) -> list[Chunk]:
        """全部含嵌入的块，按 chunk id 升序——与 all_embeddings 严格对齐。"""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, paper_id, seq, page, text FROM chunks WHERE embedding IS NOT NULL ORDER BY id"
            ).fetchall()
        return [
            Chunk(id=r["id"], paper_id=r["paper_id"], seq=r["seq"], page=r["page"], text=r["text"])
            for r in rows
        ]

    def search_snapshot(self) -> SearchSnapshot:
        """在同一 SQLite 读事务中返回完整且严格对齐的检索语料。

        返回值不再依赖数据库连接，调用方可在释放锁后安全地构建 BM25、
        计算向量并组装命中结果。``revision`` 可用作进程内检索缓存的失效键。
        """
        with self._lock:
            self._conn.execute("BEGIN")
            try:
                meta_rows = self._conn.execute(
                    "SELECT key, value FROM meta WHERE key IN ('embed_model', 'index_revision')"
                ).fetchall()
                rows = self._conn.execute(
                    """SELECT c.id AS chunk_id, c.paper_id, c.page, c.text, c.embedding,
                              p.title, p.authors, p.year, p.path
                       FROM chunks c JOIN papers p ON p.id = c.paper_id
                       WHERE c.embedding IS NOT NULL
                       ORDER BY c.id"""
                ).fetchall()
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

        meta = {r["key"]: r["value"] for r in meta_rows}
        items = tuple(
            SearchCorpusItem(
                chunk_id=r["chunk_id"],
                paper_id=r["paper_id"],
                title=r["title"],
                authors=tuple(json.loads(r["authors"])),
                year=r["year"],
                path=r["path"],
                page=r["page"],
                text=r["text"],
            )
            for r in rows
        )
        if rows:
            try:
                embeddings = np.vstack(
                    [np.frombuffer(r["embedding"], dtype=np.float32) for r in rows]
                )
            except ValueError as exc:
                raise RuntimeError("索引中的嵌入维度不一致，请使用 --force 全量重建索引") from exc
        else:
            embeddings = np.zeros((0, 0), dtype=np.float32)
        embeddings.setflags(write=False)
        try:
            revision = int(meta.get("index_revision", "0"))
        except (TypeError, ValueError):
            revision = 0
        return SearchSnapshot(
            items=items,
            embeddings=embeddings,
            embed_model=meta.get("embed_model"),
            revision=revision,
        )

    def get_chunks(self, ids: list[int]) -> list[SearchHit]:
        """按 ids 顺序返回命中块，并 join papers 补全元数据。"""
        if not ids:
            return []
        marks = ",".join("?" * len(ids))
        with self._lock:
            rows = self._conn.execute(
                f"""SELECT c.id AS chunk_id, c.paper_id, c.page, c.text,
                           p.title, p.authors, p.year, p.path
                    FROM chunks c JOIN papers p ON p.id = c.paper_id
                    WHERE c.id IN ({marks})""",
                ids,
            ).fetchall()
        by_id = {r["chunk_id"]: r for r in rows}
        hits = []
        for cid in ids:
            r = by_id.get(cid)
            if r is None:
                continue
            hits.append(
                SearchHit(
                    chunk_id=r["chunk_id"],
                    paper_id=r["paper_id"],
                    title=r["title"],
                    authors=json.loads(r["authors"]),
                    year=r["year"],
                    path=r["path"],
                    page=r["page"],
                    text=r["text"],
                    score=0.0,
                )
            )
        return hits

    # ---------- evidence ----------

    def evidence_from_chunk(self, chunk_id: int) -> Evidence:
        """从当前索引分块生成稳定、尚未固定的证据快照。"""

        with self._lock:
            row = self._conn.execute(
                """SELECT c.id AS chunk_id, c.paper_id, c.seq, c.page, c.text,
                          p.path, p.sha256, p.title, p.authors, p.year
                   FROM chunks c JOIN papers p ON p.id = c.paper_id
                   WHERE c.id=?""",
                (chunk_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"分块不存在：{chunk_id}")
        text_hash = self._text_sha256(row["text"])
        source_hash = self._evidence_source_hash(
            row["sha256"], row["seq"], row["page"], text_hash
        )
        return Evidence(
            id=f"ev_{source_hash}",
            paper_id=row["paper_id"],
            chunk_id=row["chunk_id"],
            source_hash=source_hash,
            paper_sha256=row["sha256"],
            chunk_text_sha256=text_hash,
            title=row["title"],
            authors=tuple(json.loads(row["authors"])),
            year=row["year"],
            path=row["path"],
            page=row["page"],
            chunk_seq=row["seq"],
            text=row["text"],
        )

    def pin_evidence(
        self,
        evidence: Evidence | int,
        annotation: Optional[str] = None,
    ) -> Evidence:
        """持久化证据及批注；同一来源重复固定只更新快照和批注。"""

        if isinstance(evidence, int):
            evidence = self.evidence_from_chunk(evidence)
        if not isinstance(evidence, Evidence):
            raise TypeError("evidence 必须是 Evidence 或 chunk id")
        text_hash = self._text_sha256(evidence.text)
        source_hash = self._evidence_source_hash(
            evidence.paper_sha256,
            evidence.chunk_seq,
            evidence.page,
            text_hash,
        )
        if (
            evidence.id != f"ev_{source_hash}"
            or evidence.source_hash != source_hash
            or evidence.chunk_text_sha256 != text_hash
        ):
            raise ValueError("Evidence 的 id/hash 与来源快照不一致")
        annotation_value = evidence.annotation if annotation is None else str(annotation)
        pinned_at = evidence.pinned_at or now_iso()
        with self._lock, self._conn:
            self._conn.execute(
                """INSERT INTO evidence (
                       id, paper_id, chunk_id, source_hash, paper_sha256,
                       chunk_text_sha256, title, authors, year, path, page,
                       chunk_seq, text, annotation, pinned_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET
                       paper_id=excluded.paper_id,
                       chunk_id=excluded.chunk_id,
                       source_hash=excluded.source_hash,
                       paper_sha256=excluded.paper_sha256,
                       chunk_text_sha256=excluded.chunk_text_sha256,
                       title=excluded.title,
                       authors=excluded.authors,
                       year=excluded.year,
                       path=excluded.path,
                       page=excluded.page,
                       chunk_seq=excluded.chunk_seq,
                       text=excluded.text,
                       annotation=excluded.annotation""",
                (
                    evidence.id,
                    evidence.paper_id,
                    evidence.chunk_id,
                    evidence.source_hash,
                    evidence.paper_sha256,
                    evidence.chunk_text_sha256,
                    evidence.title,
                    json.dumps(list(evidence.authors), ensure_ascii=False),
                    evidence.year,
                    evidence.path,
                    evidence.page,
                    evidence.chunk_seq,
                    evidence.text,
                    annotation_value,
                    pinned_at,
                ),
            )
            row = self._conn.execute(
                "SELECT * FROM evidence WHERE id=?", (evidence.id,)
            ).fetchone()
            if row is None:  # pragma: no cover - INSERT 后仅数据库损坏时可能发生
                raise RuntimeError("证据写入后无法读取")
            pinned = self._evidence_from_row_locked(row)
        return pinned

    def get_evidence(self, evidence_id: str) -> Optional[Evidence]:
        """读取证据，并根据当前索引实时计算 ``stale`` 状态。"""

        with self._lock:
            self._conn.execute("BEGIN")
            try:
                row = self._conn.execute(
                    "SELECT * FROM evidence WHERE id=?", (evidence_id,)
                ).fetchone()
                evidence = self._evidence_from_row_locked(row) if row else None
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
        return evidence

    def list_evidence(self, limit: int = 100, offset: int = 0) -> list[Evidence]:
        """按固定时间倒序列出证据，并逐条检查当前来源。"""

        if limit < 0 or offset < 0:
            raise ValueError("limit 和 offset 必须是非负整数")
        with self._lock:
            self._conn.execute("BEGIN")
            try:
                rows = self._conn.execute(
                    """SELECT * FROM evidence
                       ORDER BY pinned_at DESC, id
                       LIMIT ? OFFSET ?""",
                    (limit, offset),
                ).fetchall()
                evidence = [self._evidence_from_row_locked(row) for row in rows]
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
        return evidence

    # ---------- agent runs / events ----------

    def create_agent_run(
        self,
        objective: str,
        plan: Any = None,
        budget: Any = None,
        *,
        status: str = "proposed",
        run_id: Optional[str] = None,
    ) -> AgentRunRecord:
        """新建可恢复的 Agent 运行。"""

        if not objective.strip():
            raise ValueError("objective 不能为空")
        self._validate_agent_run_status(status, parameter="status")
        run_id = run_id or f"run_{uuid.uuid4().hex}"
        created_at = now_iso()
        with self._lock, self._conn:
            self._conn.execute(
                """INSERT INTO agent_runs
                       (id, objective, status, plan, budget, error, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, NULL, ?, ?)""",
                (
                    run_id,
                    objective,
                    status,
                    self._json_dump_optional(plan),
                    self._json_dump_optional(budget),
                    created_at,
                    created_at,
                ),
            )
            row = self._conn.execute(
                "SELECT * FROM agent_runs WHERE id=?", (run_id,)
            ).fetchone()
            if row is None:  # pragma: no cover - INSERT 后仅数据库损坏时可能发生
                raise RuntimeError("Agent Run 写入后无法读取")
            record = self._agent_run_from_row(row)
        return record

    def get_agent_run(self, run_id: str) -> Optional[AgentRunRecord]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM agent_runs WHERE id=?", (run_id,)
            ).fetchone()
        return self._agent_run_from_row(row) if row else None

    def transition_agent_run(
        self,
        run_id: str,
        to_status: str,
        expected_status: Optional[str] = None,
        error: Optional[str] = None,
    ) -> AgentRunRecord:
        """原子转换运行状态；提供 expected_status 时执行 compare-and-swap。"""

        self._validate_agent_run_status(to_status, parameter="to_status")
        if expected_status is not None:
            self._validate_agent_run_status(expected_status, parameter="expected_status")
        updated_at = now_iso()
        with self._lock, self._conn:
            if expected_status is None:
                cursor = self._conn.execute(
                    """UPDATE agent_runs SET status=?, error=?, updated_at=?
                       WHERE id=?""",
                    (to_status, error, updated_at, run_id),
                )
            else:
                cursor = self._conn.execute(
                    """UPDATE agent_runs SET status=?, error=?, updated_at=?
                       WHERE id=? AND status=?""",
                    (to_status, error, updated_at, run_id, expected_status),
                )
            if not cursor.rowcount:
                row = self._conn.execute(
                    "SELECT status FROM agent_runs WHERE id=?", (run_id,)
                ).fetchone()
                if row is None:
                    raise KeyError(f"Agent Run 不存在：{run_id}")
                raise AgentRunStatusConflictError(
                    f"Agent Run {run_id} 状态冲突："
                    f"期望 {expected_status!r}，当前 {row['status']!r}"
                )
            row = self._conn.execute(
                "SELECT * FROM agent_runs WHERE id=?", (run_id,)
            ).fetchone()
            if row is None:  # pragma: no cover
                raise RuntimeError("Agent Run 状态更新后无法读取")
            record = self._agent_run_from_row(row)
        return record

    def append_agent_event(
        self,
        run_id: str,
        event_type: str,
        payload: Any = None,
    ) -> AgentEventRecord:
        """为运行追加一个事务内分配序号的事件。"""

        if not event_type.strip():
            raise ValueError("event_type 不能为空")
        created_at = now_iso()
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                if self._conn.execute(
                    "SELECT 1 FROM agent_runs WHERE id=?", (run_id,)
                ).fetchone() is None:
                    raise KeyError(f"Agent Run 不存在：{run_id}")
                seq_row = self._conn.execute(
                    "SELECT COALESCE(MAX(seq), 0) + 1 AS seq FROM agent_events WHERE run_id=?",
                    (run_id,),
                ).fetchone()
                cursor = self._conn.execute(
                    """INSERT INTO agent_events
                           (run_id, seq, event_type, payload, created_at)
                       VALUES (?, ?, ?, ?, ?)""",
                    (
                        run_id,
                        seq_row["seq"],
                        event_type,
                        self._json_dump_optional(payload),
                        created_at,
                    ),
                )
                event_id = int(cursor.lastrowid)
                row = self._conn.execute(
                    "SELECT * FROM agent_events WHERE id=?", (event_id,)
                ).fetchone()
                if row is None:  # pragma: no cover
                    raise RuntimeError("Agent Event 写入后无法读取")
                event = self._agent_event_from_row(row)
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
        return event

    def list_agent_events(
        self,
        run_id: str,
        after_seq: int = 0,
        limit: int = 1000,
    ) -> list[AgentEventRecord]:
        if after_seq < 0 or limit < 0:
            raise ValueError("after_seq 和 limit 必须是非负整数")
        with self._lock:
            rows = self._conn.execute(
                """SELECT * FROM agent_events
                   WHERE run_id=? AND seq>?
                   ORDER BY seq LIMIT ?""",
                (run_id, after_seq, limit),
            ).fetchall()
        return [self._agent_event_from_row(row) for row in rows]

    def list_agent_runs(
        self,
        limit: int = 50,
        offset: int = 0,
    ) -> list[AgentRunRecord]:
        """按创建时间倒序浏览 Agent run（Web 审计侧栏）。"""
        if limit < 0 or offset < 0:
            raise ValueError("limit 和 offset 必须是非负整数")
        with self._lock:
            rows = self._conn.execute(
                """SELECT * FROM agent_runs
                   ORDER BY created_at DESC, id DESC
                   LIMIT ? OFFSET ?""",
                (limit, offset),
            ).fetchall()
        return [self._agent_run_from_row(row) for row in rows]

    # ---------- meta / stats ----------

    def meta_get(self, key: str) -> Optional[str]:
        with self._lock:
            row = self._conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row["value"] if row else None

    def meta_set(self, key: str, value: str) -> None:
        with self._lock, self._conn:
            changed = self._set_meta_locked(key, value)
            if key == "embed_model" and changed:
                self._bump_revision_locked()

    def search_cache_key(self) -> tuple[str, int, Optional[str]]:
        """以单条查询返回当前检索缓存键。"""
        with self._lock:
            row = self._conn.execute(
                """SELECT
                       MAX(CASE WHEN key='index_revision' THEN value END) AS revision,
                       MAX(CASE WHEN key='embed_model' THEN value END) AS embed_model
                   FROM meta WHERE key IN ('index_revision', 'embed_model')"""
            ).fetchone()
        revision = self._parse_revision(row["revision"])
        return self._db_identity, revision, row["embed_model"]

    @property
    def revision(self) -> int:
        """当前搜索索引版本；搜索可复用缓存应以此值作为失效依据。"""
        return self.search_cache_key()[1]

    def stats(self) -> tuple[int, int]:
        with self._lock:
            row = self._conn.execute(
                "SELECT (SELECT COUNT(*) FROM papers) AS p, (SELECT COUNT(*) FROM chunks) AS c"
            ).fetchone()
        return row["p"], row["c"]

    # ---------- helpers ----------

    def _bump_revision_locked(self) -> None:
        self._conn.execute(
            """INSERT INTO meta (key, value) VALUES ('index_revision', '1')
               ON CONFLICT(key) DO UPDATE
               SET value = CAST(COALESCE(value, '0') AS INTEGER) + 1"""
        )

    def _begin_immediate_locked(self) -> None:
        try:
            self._conn.execute("BEGIN IMMEDIATE")
        except sqlite3.OperationalError as exc:
            if "locked" in str(exc).lower() or "busy" in str(exc).lower():
                raise RevisionConflictError(
                    "索引库正被其他进程修改；本次写入已取消，请稍后重试。"
                ) from exc
            raise

    def _assert_revision_locked(self, expected_revision: int) -> None:
        row = self._conn.execute(
            "SELECT value FROM meta WHERE key='index_revision'"
        ).fetchone()
        actual_revision = self._parse_revision(row["value"] if row else None)
        if actual_revision != expected_revision:
            raise RevisionConflictError(
                "索引在预处理期间已被其他进程修改"
                f"（期望版本 {expected_revision}，当前版本 {actual_revision}）；"
                "本次写入已取消，请重试。"
            )

    def _set_meta_locked(self, key: str, value: str) -> bool:
        cursor = self._conn.execute(
            """INSERT INTO meta (key, value) VALUES (?, ?)
               ON CONFLICT(key) DO UPDATE SET value=excluded.value
               WHERE meta.value IS NOT excluded.value""",
            (key, value),
        )
        return bool(cursor.rowcount)

    @staticmethod
    def _parse_revision(value: object) -> int:
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _validate_agent_run_status(status: str, *, parameter: str) -> None:
        if status not in _AGENT_RUN_STATUSES:
            allowed = ", ".join(sorted(_AGENT_RUN_STATUSES))
            raise ValueError(f"{parameter} 必须是以下状态之一：{allowed}")

    @staticmethod
    def _chunk_from_row(row: sqlite3.Row, include_embedding: bool = False) -> Chunk:
        embedding = None
        if include_embedding and "embedding" in row.keys() and row["embedding"] is not None:
            embedding = np.frombuffer(row["embedding"], dtype=np.float32)
        return Chunk(
            id=row["id"],
            paper_id=row["paper_id"],
            seq=row["seq"],
            page=row["page"],
            text=row["text"],
            embedding=embedding,
        )

    @staticmethod
    def _text_sha256(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def _evidence_source_hash(
        paper_sha256: str,
        chunk_seq: int,
        page: int,
        chunk_text_sha256: str,
    ) -> str:
        source = json.dumps(
            [paper_sha256, int(chunk_seq), int(page), chunk_text_sha256],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return hashlib.sha256(source.encode("utf-8")).hexdigest()

    def _evidence_from_row_locked(self, row: sqlite3.Row) -> Evidence:
        stale = False
        stale_reason: Optional[str] = None
        current_paper = self._conn.execute(
            "SELECT id, sha256 FROM papers WHERE path=?", (row["path"],)
        ).fetchone()
        current_chunk = None
        if current_paper is None:
            stale = True
            stale_reason = "论文已从当前索引移除"
        elif current_paper["sha256"] != row["paper_sha256"]:
            stale = True
            stale_reason = "论文内容哈希已变化"
        else:
            current_chunk = self._conn.execute(
                """SELECT id, paper_id, seq, page, text
                   FROM chunks WHERE paper_id=? AND seq=?""",
                (current_paper["id"], row["chunk_seq"]),
            ).fetchone()
            if current_chunk is None:
                stale = True
                stale_reason = "来源分块已从当前索引移除"
            else:
                current_text_hash = self._text_sha256(current_chunk["text"])
                current_source_hash = self._evidence_source_hash(
                    current_paper["sha256"],
                    current_chunk["seq"],
                    current_chunk["page"],
                    current_text_hash,
                )
                if (
                    current_chunk["page"] != row["page"]
                    or current_text_hash != row["chunk_text_sha256"]
                    or current_source_hash != row["source_hash"]
                ):
                    stale = True
                    stale_reason = "来源分块内容或位置已变化"

        return Evidence(
            id=row["id"],
            paper_id=(
                current_paper["id"]
                if not stale and current_paper is not None
                else row["paper_id"]
            ),
            chunk_id=(
                current_chunk["id"]
                if not stale and current_chunk is not None
                else row["chunk_id"]
            ),
            source_hash=row["source_hash"],
            paper_sha256=row["paper_sha256"],
            chunk_text_sha256=row["chunk_text_sha256"],
            title=row["title"],
            authors=tuple(json.loads(row["authors"])),
            year=row["year"],
            path=row["path"],
            page=row["page"],
            chunk_seq=row["chunk_seq"],
            text=row["text"],
            annotation=row["annotation"],
            pinned_at=row["pinned_at"],
            stale=stale,
            stale_reason=stale_reason,
        )

    @staticmethod
    def _json_dump_optional(value: Any) -> Optional[str]:
        if value is None:
            return None
        to_dict = getattr(value, "to_dict", None)
        if callable(to_dict):
            value = to_dict()
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _json_load_optional(value: Optional[str]) -> Any:
        return None if value is None else json.loads(value)

    @classmethod
    def _agent_run_from_row(cls, row: sqlite3.Row) -> AgentRunRecord:
        return AgentRunRecord(
            id=row["id"],
            objective=row["objective"],
            status=row["status"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            plan=cls._json_load_optional(row["plan"]),
            budget=cls._json_load_optional(row["budget"]),
            error=row["error"],
        )

    @classmethod
    def _agent_event_from_row(cls, row: sqlite3.Row) -> AgentEventRecord:
        return AgentEventRecord(
            id=row["id"],
            run_id=row["run_id"],
            seq=row["seq"],
            event_type=row["event_type"],
            created_at=row["created_at"],
            payload=cls._json_load_optional(row["payload"]),
        )

    @staticmethod
    def _paper_from_row(row: sqlite3.Row) -> Paper:
        return Paper(
            id=row["id"],
            path=row["path"],
            sha256=row["sha256"],
            title=row["title"],
            authors=json.loads(row["authors"]),
            year=row["year"],
            page_count=row["page_count"],
            has_text=bool(row["has_text"]),
            indexed_at=row["indexed_at"],
        )


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")
