"""研究 repository 共用的 SQLite 连接与事务边界。"""

from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

from .migrations import MigrationReport, migrate_schema


class RecordVersionConflictError(RuntimeError):
    """Repository compare-and-swap 检测到旧版本写入。"""


class SQLiteRepository:
    """每个 repository 实例拥有独立连接；跨进程写入用 SQLite CAS 协调。"""

    def __init__(self, db_path: str | Path):
        db_value = str(db_path)
        self._db_path = (
            None
            if db_value == ":memory:"
            else Path(db_value).expanduser().resolve(strict=False)
        )
        self._conn = sqlite3.connect(db_value, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        try:
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.execute("PRAGMA busy_timeout=5000")
            self._migration_report = migrate_schema(
                self._conn,
                db_path=self._db_path,
            )
            self._conn.execute("PRAGMA journal_mode=WAL")
        except Exception:
            self._conn.close()
            raise

    @property
    def db_path(self) -> Optional[Path]:
        return self._db_path

    @property
    def migration_report(self) -> MigrationReport:
        return self._migration_report

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @contextmanager
    def _transaction(self, *, immediate: bool = False) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
            try:
                yield self._conn
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
