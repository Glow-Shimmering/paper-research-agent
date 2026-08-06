"""SQLite 存储层：论文、分块、meta。跨线程安全（Web 线程池 + reindex 线程）。"""
import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

from .models import Chunk, Paper, SearchHit

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


class Store:
    def __init__(self, db_path: str | Path):
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()
            if self.meta_get("schema_version") is None:
                self.meta_set("schema_version", "1")

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ---------- papers ----------

    def upsert_paper(self, paper: Paper, chunks: Optional[list[Chunk]] = None) -> int:
        """按 path 幂等写入；chunks 非空时同事务替换该论文所有分块。"""
        with self._lock, self._conn:
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
            # 注意：DO UPDATE 分支不更新连接级 last_insert_rowid，必须按 path 回查
            row = self._conn.execute(
                "SELECT id FROM papers WHERE path=?", (paper.path,)
            ).fetchone()
            paper_id = row["id"]
            if chunks is not None:
                self._replace_chunks_locked(paper_id, chunks)
        return paper_id

    def delete_paper(self, paper_id: int) -> None:
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM papers WHERE id=?", (paper_id,))

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

    def iter_papers(self) -> Iterator[Paper]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM papers ORDER BY id").fetchall()
        return iter([self._paper_from_row(r) for r in rows])

    # ---------- chunks ----------

    def replace_chunks(self, paper_id: int, chunks: list[Chunk]) -> None:
        with self._lock, self._conn:
            self._replace_chunks_locked(paper_id, chunks)

    def _replace_chunks_locked(self, paper_id: int, chunks: list[Chunk]) -> None:
        self._conn.execute("DELETE FROM chunks WHERE paper_id=?", (paper_id,))
        for c in chunks:
            blob = c.embedding.astype(np.float32).tobytes() if c.embedding is not None else None
            self._conn.execute(
                "INSERT INTO chunks (paper_id, seq, page, text, embedding) VALUES (?, ?, ?, ?, ?)",
                (paper_id, c.seq, c.page, c.text, blob),
            )

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
            self._conn.execute(
                "INSERT INTO meta (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )

    def stats(self) -> tuple[int, int]:
        with self._lock:
            row = self._conn.execute(
                "SELECT (SELECT COUNT(*) FROM papers) AS p, (SELECT COUNT(*) FROM chunks) AS c"
            ).fetchone()
        return row["p"], row["c"]

    # ---------- helpers ----------

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
