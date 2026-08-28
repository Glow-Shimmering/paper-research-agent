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
_AGENT_MESSAGE_ROLES = frozenset({"system", "user", "assistant", "tool"})
_MAX_AGENT_SESSION_ID_CHARS = 128


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

    def set_paper_document_metadata(
        self,
        paper_id: int,
        *,
        source_kind: str,
        canonical_uri: Optional[str],
        locator: Any,
    ) -> Paper:
        """更新通用 document locator；发生变化时使检索 snapshot cache 失效。"""

        if source_kind not in {"pdf", "web"}:
            raise ValueError("source_kind 必须是 pdf 或 web")
        try:
            locator_json = json.dumps(
                locator,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("locator 必须可序列化为 JSON") from exc
        with self._lock, self._conn:
            row = self._conn.execute(
                "SELECT * FROM papers WHERE id=?", (paper_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"索引文档不存在：{paper_id}")
            if (
                row["source_kind"] != source_kind
                or row["canonical_uri"] != canonical_uri
                or row["locator"] != locator_json
            ):
                self._conn.execute(
                    """
                    UPDATE papers SET source_kind=?, canonical_uri=?, locator=?
                    WHERE id=?
                    """,
                    (source_kind, canonical_uri, locator_json, paper_id),
                )
                self._bump_revision_locked()
                row = self._conn.execute(
                    "SELECT * FROM papers WHERE id=?", (paper_id,)
                ).fetchone()
        return self._paper_from_row(row)

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
        if paper.source_kind not in {"pdf", "web"}:
            raise ValueError("paper source_kind 必须是 pdf 或 web")
        try:
            locator_json = json.dumps(
                paper.locator,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("paper locator 必须可序列化为 JSON") from exc
        self._conn.execute(
            """INSERT INTO papers (
                   path, sha256, title, authors, year, page_count, has_text,
                   indexed_at, source_kind, canonical_uri, locator
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(path) DO UPDATE SET
                   sha256=excluded.sha256, title=excluded.title, authors=excluded.authors,
                   year=excluded.year, page_count=excluded.page_count,
                   has_text=excluded.has_text, indexed_at=excluded.indexed_at,
                   source_kind=excluded.source_kind,
                   canonical_uri=excluded.canonical_uri,
                   locator=excluded.locator""",
            (
                paper.path,
                paper.sha256,
                paper.title,
                json.dumps(paper.authors, ensure_ascii=False),
                paper.year,
                paper.page_count,
                int(paper.has_text),
                paper.indexed_at,
                paper.source_kind,
                paper.canonical_uri,
                locator_json,
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
                              p.title, p.authors, p.year, p.path, p.source_kind,
                              p.canonical_uri, p.locator
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
                source_kind=r["source_kind"],
                canonical_uri=r["canonical_uri"],
                locator=json.loads(r["locator"]),
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
                           p.title, p.authors, p.year, p.path, p.source_kind,
                           p.canonical_uri, p.locator
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
                    source_kind=r["source_kind"],
                    canonical_uri=r["canonical_uri"],
                    locator=json.loads(r["locator"]),
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
        project_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> AgentRunRecord:
        """新建可恢复的 Agent 运行。"""

        if not objective.strip():
            raise ValueError("objective 不能为空")
        self._validate_agent_run_status(status, parameter="status")
        run_id = run_id or f"run_{uuid.uuid4().hex}"
        created_at = now_iso()
        with self._lock, self._conn:
            if session_id is not None:
                session_id = self._validate_agent_session_id(session_id)
                session = self._conn.execute(
                    "SELECT project_id FROM agent_sessions WHERE id=?", (session_id,)
                ).fetchone()
                if session is None:
                    raise KeyError(f"Agent session 不存在：{session_id}")
                if session["project_id"] != project_id:
                    raise ValueError("Agent run 的 project 与 session 不一致")
            elif project_id is not None and self._conn.execute(
                "SELECT 1 FROM research_projects WHERE id=?", (project_id,)
            ).fetchone() is None:
                raise KeyError(f"研究项目不存在：{project_id}")
            self._conn.execute(
                """INSERT INTO agent_runs
                       (id, objective, status, plan, budget, error, created_at, updated_at,
                        project_id, session_id)
                   VALUES (?, ?, ?, ?, ?, NULL, ?, ?, ?, ?)""",
                (
                    run_id,
                    objective,
                    status,
                    self._json_dump_optional(plan),
                    self._json_dump_optional(budget),
                    created_at,
                    created_at,
                    project_id,
                    session_id,
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
        *,
        project_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> list[AgentRunRecord]:
        """按创建时间倒序浏览 Agent run（Web 审计侧栏）。"""
        if limit < 0 or offset < 0:
            raise ValueError("limit 和 offset 必须是非负整数")
        where: list[str] = []
        params: list[Any] = []
        if project_id is not None:
            where.append("project_id=?")
            params.append(project_id)
        if session_id is not None:
            where.append("session_id=?")
            params.append(self._validate_agent_session_id(session_id))
        clause = f"WHERE {' AND '.join(where)}" if where else ""
        with self._lock:
            rows = self._conn.execute(
                f"""SELECT * FROM agent_runs {clause}
                   ORDER BY created_at DESC, id DESC
                   LIMIT ? OFFSET ?""",
                (*params, limit, offset),
            ).fetchall()
        return [self._agent_run_from_row(row) for row in rows]

    # ---------- agent sessions / transcript ----------

    def ensure_agent_session(
        self,
        session_id: str,
        *,
        project_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """创建会话壳或验证既有 project 绑定；绑定一旦产生便不可改写。"""

        normalized_id = self._validate_agent_session_id(session_id)
        timestamp = now_iso()
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                if project_id is not None and self._conn.execute(
                    "SELECT 1 FROM research_projects WHERE id=?", (project_id,)
                ).fetchone() is None:
                    raise KeyError(f"研究项目不存在：{project_id}")
                row = self._conn.execute(
                    "SELECT * FROM agent_sessions WHERE id=?", (normalized_id,)
                ).fetchone()
                if row is None:
                    self._conn.execute(
                        """INSERT INTO agent_sessions
                               (id, project_id, title, status, version, created_at, updated_at)
                           VALUES (?, ?, '', 'active', 1, ?, ?)""",
                        (normalized_id, project_id, timestamp, timestamp),
                    )
                elif project_id is not None and row["project_id"] != project_id:
                    if row["project_id"] is not None or self._session_has_history_locked(
                        normalized_id
                    ):
                        raise ValueError("Agent session 已绑定其他 project，不能改写")
                    self._conn.execute(
                        """UPDATE agent_sessions
                           SET project_id=?, version=version+1, updated_at=? WHERE id=?""",
                        (project_id, timestamp, normalized_id),
                    )
                row = self._conn.execute(
                    "SELECT * FROM agent_sessions WHERE id=?", (normalized_id,)
                ).fetchone()
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
        return dict(row)

    def load_agent_session_state(self, session_id: str) -> Optional[dict[str, Any]]:
        """读取 project 绑定、完整 transcript 与唯一未决确认票据。"""

        normalized_id = self._validate_agent_session_id(session_id)
        with self._lock:
            session = self._conn.execute(
                "SELECT * FROM agent_sessions WHERE id=?", (normalized_id,)
            ).fetchone()
            if session is None:
                return None
            pending = self._conn.execute(
                """SELECT * FROM pending_actions
                   WHERE session_id=? AND status='pending'
                   ORDER BY created_at DESC, id DESC LIMIT 1""",
                (normalized_id,),
            ).fetchone()
        result = dict(session)
        result["messages"] = self.load_agent_messages(normalized_id)
        result["pending_action"] = self._pending_action_from_row(pending) if pending else None
        return result

    def load_agent_messages(self, session_id: str) -> list[dict[str, Any]]:
        """读取一个已完成 Web 会话的完整模型消息。

        ``agent_messages.content`` 保存整条 OpenAI-compatible 消息的 JSON，
        因而 assistant ``tool_calls`` 和 tool ``tool_call_id`` 可无损恢复。
        对早期仅保存纯文本的记录保留兼容读取路径。
        """

        normalized_id = self._validate_agent_session_id(session_id)
        with self._lock:
            rows = self._conn.execute(
                """SELECT role, content FROM agent_messages
                   WHERE session_id=? ORDER BY seq""",
                (normalized_id,),
            ).fetchall()
        messages: list[dict[str, Any]] = []
        for row in rows:
            try:
                decoded = json.loads(row["content"])
            except (TypeError, json.JSONDecodeError):
                decoded = None
            if isinstance(decoded, dict) and decoded.get("role") == row["role"]:
                messages.append(decoded)
            else:
                messages.append({"role": row["role"], "content": row["content"]})
        return messages

    def save_agent_messages(
        self,
        session_id: str,
        messages: list[dict[str, Any]],
    ) -> None:
        """兼容入口：原子替换 transcript，保留既有 project 绑定。"""

        self.save_agent_session_state(session_id, messages)

    def save_agent_session_state(
        self,
        session_id: str,
        messages: list[dict[str, Any]],
        *,
        project_id: Optional[str] = None,
        pending_action: Optional[dict[str, Any]] = None,
    ) -> None:
        """原子保存 transcript，并可同时冻结一张待确认工具票据。"""

        normalized_id = self._validate_agent_session_id(session_id)
        serialized: list[tuple[str, str]] = []
        for message in messages:
            if not isinstance(message, dict):
                raise ValueError("Agent message 必须是字典")
            role = str(message.get("role") or "")
            if role not in _AGENT_MESSAGE_ROLES:
                raise ValueError(f"Agent message role 不合法：{role!r}")
            try:
                content = json.dumps(
                    message,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            except (TypeError, ValueError) as exc:
                raise ValueError("Agent message 必须可 JSON 序列化") from exc
            serialized.append((role, content))

        pending_values: Optional[dict[str, Any]] = None
        if pending_action is not None:
            pending_values = self._validate_pending_action(pending_action)
        timestamp = now_iso()
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                if project_id is not None and self._conn.execute(
                    "SELECT 1 FROM research_projects WHERE id=?", (project_id,)
                ).fetchone() is None:
                    raise KeyError(f"研究项目不存在：{project_id}")
                existing = self._conn.execute(
                    "SELECT project_id FROM agent_sessions WHERE id=?", (normalized_id,)
                ).fetchone()
                if existing is not None and project_id is not None and existing["project_id"] != project_id:
                    if existing["project_id"] is not None or self._session_has_history_locked(
                        normalized_id
                    ):
                        raise ValueError("Agent session 已绑定其他 project，不能改写")
                self._conn.execute(
                    """INSERT INTO agent_sessions
                           (id, project_id, title, status, version, created_at, updated_at)
                       VALUES (?, ?, '', 'active', 1, ?, ?)
                       ON CONFLICT(id) DO UPDATE SET
                           project_id=COALESCE(agent_sessions.project_id, excluded.project_id),
                           status='active', version=agent_sessions.version + 1,
                           updated_at=excluded.updated_at""",
                    (normalized_id, project_id, timestamp, timestamp),
                )
                self._conn.execute(
                    "DELETE FROM agent_messages WHERE session_id=?",
                    (normalized_id,),
                )
                self._conn.executemany(
                    """INSERT INTO agent_messages
                           (id, session_id, seq, role, content, run_id, created_at)
                       VALUES (?, ?, ?, ?, ?, NULL, ?)""",
                    [
                        (
                            f"msg_{uuid.uuid4().hex}",
                            normalized_id,
                            seq,
                            role,
                            content,
                            timestamp,
                        )
                        for seq, (role, content) in enumerate(serialized, start=1)
                    ],
                )
                if pending_values is not None:
                    run = self._conn.execute(
                        "SELECT project_id, session_id FROM agent_runs WHERE id=?",
                        (pending_values["run_id"],),
                    ).fetchone()
                    if run is None or run["session_id"] != normalized_id:
                        raise ValueError("待确认动作必须绑定当前 session 的 Agent run")
                    bound_project = self._conn.execute(
                        "SELECT project_id FROM agent_sessions WHERE id=?", (normalized_id,)
                    ).fetchone()["project_id"]
                    if run["project_id"] != bound_project:
                        raise ValueError("待确认动作的 project 与 session 不一致")
                    self._conn.execute(
                        """INSERT INTO pending_actions(
                               id, session_id, run_id, tool_call_id, tool_name, tool_version,
                               arguments, arguments_sha256, confirmation, result, error,
                               status, version, created_at, updated_at, expires_at, resolved_at
                           ) VALUES (?, ?, ?, ?, ?, '1', ?, ?, ?, NULL, NULL,
                                     'pending', 1, ?, ?, NULL, NULL)
                           ON CONFLICT(id) DO UPDATE SET
                               arguments=excluded.arguments,
                               arguments_sha256=excluded.arguments_sha256,
                               confirmation=excluded.confirmation,
                               version=pending_actions.version+1,
                               updated_at=excluded.updated_at
                           WHERE pending_actions.status='pending'""",
                        (
                            pending_values["action_id"], normalized_id,
                            pending_values["run_id"], pending_values["tool_call_id"],
                            pending_values["name"], pending_values["arguments"],
                            pending_values["arguments_sha256"], pending_values["confirmation"],
                            timestamp, timestamp,
                        ),
                    )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def resolve_pending_action(
        self,
        action_id: str,
        status: str,
        *,
        result: Any = None,
        error: Optional[str] = None,
    ) -> bool:
        """以 pending 状态为 CAS 前提关闭票据，避免重复执行。"""

        if status not in {"approved", "rejected", "executed", "cancelled", "expired"}:
            raise ValueError("pending action 终态不合法")
        timestamp = now_iso()
        expected_status = "approved" if status == "executed" else "pending"
        with self._lock, self._conn:
            cursor = self._conn.execute(
                """UPDATE pending_actions
                   SET status=?, result=?, error=?, version=version+1,
                       updated_at=?, resolved_at=?
                   WHERE id=? AND status=?""",
                (
                    status,
                    self._json_dump_optional(result),
                    error,
                    timestamp,
                    timestamp,
                    str(action_id),
                    expected_status,
                ),
            )
        return bool(cursor.rowcount)

    def claim_pending_action(self, action_id: str) -> bool:
        """在执行有副作用的工具前，把票据从 pending 原子认领为 approved。"""

        timestamp = now_iso()
        with self._lock, self._conn:
            cursor = self._conn.execute(
                """UPDATE pending_actions
                   SET status='approved', version=version+1, updated_at=?
                   WHERE id=? AND status='pending'""",
                (timestamp, str(action_id)),
            )
        return bool(cursor.rowcount)

    def _session_has_history_locked(self, session_id: str) -> bool:
        return self._conn.execute(
            """SELECT EXISTS(
                   SELECT 1 FROM agent_messages WHERE session_id=?
                   UNION ALL
                   SELECT 1 FROM pending_actions WHERE session_id=?
               )""",
            (session_id, session_id),
        ).fetchone()[0] == 1

    @staticmethod
    def _validate_pending_action(value: dict[str, Any]) -> dict[str, Any]:
        required = ("action_id", "name", "args", "digest", "tool_call_id", "run_id")
        if not isinstance(value, dict) or any(value.get(key) in (None, "") for key in required):
            raise ValueError("待确认动作缺少持久化绑定字段")
        args = value["args"]
        if not isinstance(args, dict):
            raise ValueError("待确认动作 args 必须是字典")
        arguments = json.dumps(args, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        confirmation = json.dumps(
            {"action_id": str(value["action_id"]), "digest": str(value["digest"])},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return {
            "action_id": str(value["action_id"]),
            "name": str(value["name"]),
            "run_id": str(value["run_id"]),
            "tool_call_id": str(value["tool_call_id"]),
            "arguments": arguments,
            "arguments_sha256": hashlib.sha256(arguments.encode("utf-8")).hexdigest(),
            "confirmation": confirmation,
        }

    @staticmethod
    def _pending_action_from_row(row: sqlite3.Row) -> dict[str, Any]:
        confirmation = json.loads(row["confirmation"])
        arguments = json.loads(row["arguments"])
        encoded = json.dumps(arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if hashlib.sha256(encoded.encode("utf-8")).hexdigest() != row["arguments_sha256"]:
            raise ValueError("待确认动作参数摘要校验失败")
        return {
            "action_id": row["id"],
            "name": row["tool_name"],
            "args": arguments,
            "digest": confirmation.get("digest"),
            "tool_call_id": row["tool_call_id"],
            "run_id": row["run_id"],
            "status": row["status"],
        }

    def delete_agent_session(self, session_id: str) -> bool:
        """删除会话及其级联 transcript；不存在时幂等返回 ``False``。"""

        normalized_id = self._validate_agent_session_id(session_id)
        with self._lock, self._conn:
            cursor = self._conn.execute(
                "DELETE FROM agent_sessions WHERE id=?",
                (normalized_id,),
            )
        return bool(cursor.rowcount)

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
    def _validate_agent_session_id(session_id: str) -> str:
        normalized = str(session_id or "").strip()
        if not normalized:
            raise ValueError("session_id 不能为空")
        if len(normalized) > _MAX_AGENT_SESSION_ID_CHARS:
            raise ValueError(
                f"session_id 不能超过 {_MAX_AGENT_SESSION_ID_CHARS} 个字符"
            )
        return normalized

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
            project_id=row["project_id"],
            session_id=row["session_id"],
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
            source_kind=row["source_kind"],
            canonical_uri=row["canonical_uri"],
            locator=json.loads(row["locator"]),
        )


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")
