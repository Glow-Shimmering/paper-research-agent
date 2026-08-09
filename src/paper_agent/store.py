"""SQLite 存储层：论文、分块、meta。跨线程安全（Web 线程池 + reindex 线程）。"""
import json
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

from .models import Chunk, Paper, SearchCorpusItem, SearchHit, SearchSnapshot

_SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    id INTEGER PRIMARY KEY,
    path TEXT UNIQUE NOT NULL,
    sha256 TEXT NOT NULL,
    title TEXT NOT NULL,
    authors TEXT NOT NULL,
    year INTEGER,
    page_count INTEGER NOT NULL,
    has_text INTEGER NOT NULL,
    indexed_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    paper_id INTEGER NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    seq INTEGER NOT NULL,
    page INTEGER NOT NULL,
    text TEXT NOT NULL,
    embedding BLOB,
    UNIQUE(paper_id, seq)
);
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""


class RevisionConflictError(RuntimeError):
    """索引预处理期间数据库已变化；本批写入必须放弃。"""


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
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.executescript(_SCHEMA)
            self._conn.execute(
                "INSERT OR IGNORE INTO meta (key, value) VALUES ('schema_version', '1')"
            )
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
        return [
            Chunk(
                id=r["id"],
                paper_id=r["paper_id"],
                seq=r["seq"],
                page=r["page"],
                text=r["text"],
                embedding=(
                    np.frombuffer(r["embedding"], dtype=np.float32) if r["embedding"] is not None else None
                )
                if include_embeddings
                else None,
            )
            for r in rows
        ]

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
