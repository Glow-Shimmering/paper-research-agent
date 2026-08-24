"""SQLite-backed research job persistence and compare-and-swap operations."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from pragent.models import Page, ResearchJob

from ._repository import RecordVersionConflictError, SQLiteRepository


class JobStateConflictError(RecordVersionConflictError):
    """Job status/version 不符合 compare-and-swap 前置条件。"""


class JobIdempotencyConflictError(RuntimeError):
    """同一 idempotency key 被用于不同的 job 请求。"""


_JOB_STATUSES = frozenset(
    {
        "queued",
        "running",
        "succeeded",
        "failed",
        "cancel_requested",
        "cancelled",
        "interrupted",
    }
)
_TERMINAL_STATUSES = frozenset({"succeeded", "failed", "cancelled"})
_ALLOWED_TRANSITIONS: dict[str, frozenset[str]] = {
    "queued": frozenset({"running", "cancel_requested", "cancelled", "failed"}),
    "running": frozenset(
        {"succeeded", "failed", "cancel_requested", "cancelled", "interrupted"}
    ),
    "cancel_requested": frozenset({"cancelled", "failed", "interrupted"}),
    "interrupted": frozenset({"queued", "cancelled", "failed"}),
    "succeeded": frozenset(),
    "failed": frozenset(),
    "cancelled": frozenset(),
}


class JobRepository(SQLiteRepository):
    def enqueue(
        self,
        job_type: str,
        payload: Any,
        *,
        project_id: Optional[str] = None,
        artifact_id: Optional[str] = None,
        priority: int = 0,
        run_after: Optional[str] = None,
        timeout_seconds: Optional[int] = None,
        max_attempts: int = 1,
        idempotent: bool = False,
        idempotency_key: Optional[str] = None,
        job_id: Optional[str] = None,
    ) -> ResearchJob:
        job_type = _required_text(job_type, "job_type")
        payload_json = _json_dump(payload)
        if not isinstance(priority, int):
            raise ValueError("priority 必须是整数")
        if not isinstance(max_attempts, int) or max_attempts < 1:
            raise ValueError("max_attempts 必须是正整数")
        if timeout_seconds is not None and (
            not isinstance(timeout_seconds, int) or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds 必须是正整数")
        if run_after is not None:
            run_after = _parse_or_now(run_after).isoformat(timespec="microseconds")
        if idempotency_key is not None:
            idempotency_key = _required_text(idempotency_key, "idempotency_key")
        if idempotent and idempotency_key is None:
            raise ValueError("idempotent job 必须提供 idempotency_key")
        job_id = job_id or f"job_{uuid.uuid4().hex}"
        now = _now_iso()
        with self._transaction(immediate=True) as connection:
            self._validate_job_scope_locked(
                connection, project_id=project_id, artifact_id=artifact_id
            )
            if idempotency_key is not None:
                existing = connection.execute(
                    "SELECT * FROM research_jobs WHERE idempotency_key=?",
                    (idempotency_key,),
                ).fetchone()
                if existing:
                    if not self._same_enqueue_request(
                        existing,
                        job_type=job_type,
                        payload_json=payload_json,
                        project_id=project_id,
                        artifact_id=artifact_id,
                        priority=priority,
                        run_after=run_after,
                        timeout_seconds=timeout_seconds,
                        max_attempts=max_attempts,
                        idempotent=idempotent,
                    ):
                        raise JobIdempotencyConflictError(
                            f"idempotency key 已用于不同任务：{idempotency_key}"
                        )
                    return self._job_from_row(existing)
            try:
                connection.execute(
                    """
                    INSERT INTO research_jobs(
                        id, project_id, artifact_id, job_type, status, payload,
                        progress_current, progress_total, attempts, max_attempts,
                        idempotent, priority, run_after, timeout_seconds,
                        idempotency_key, version, created_at, updated_at
                    ) VALUES (
                        ?, ?, ?, ?, 'queued', ?, 0, NULL, 0, ?, ?, ?, ?, ?, ?, 1, ?, ?
                    )
                    """,
                    (
                        job_id,
                        project_id,
                        artifact_id,
                        job_type,
                        payload_json,
                        max_attempts,
                        int(idempotent),
                        priority,
                        run_after,
                        timeout_seconds,
                        idempotency_key,
                        now,
                        now,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                self._translate_foreign_key_error_locked(
                    connection, project_id=project_id, artifact_id=artifact_id
                )
                raise exc
            row = connection.execute(
                "SELECT * FROM research_jobs WHERE id=?", (job_id,)
            ).fetchone()
        return self._job_from_row(row)

    def get(self, job_id: str) -> Optional[ResearchJob]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM research_jobs WHERE id=?", (job_id,)
            ).fetchone()
        return self._job_from_row(row) if row else None

    def list(
        self,
        *,
        project_id: Optional[str] = None,
        status: Optional[str] = None,
        job_type: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Page[ResearchJob]:
        limit, offset = _validate_page(limit, offset)
        where: list[str] = []
        params: list[Any] = []
        if project_id is not None:
            where.append("project_id=?")
            params.append(project_id)
        if status is not None:
            _validate_status(status)
            where.append("status=?")
            params.append(status)
        if job_type is not None:
            where.append("job_type=?")
            params.append(job_type)
        clause = f"WHERE {' AND '.join(where)}" if where else ""
        with self._lock:
            total = int(
                self._conn.execute(
                    f"SELECT COUNT(*) FROM research_jobs {clause}", params
                ).fetchone()[0]
            )
            rows = self._conn.execute(
                f"""
                SELECT * FROM research_jobs {clause}
                ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?
                """,
                (*params, limit, offset),
            ).fetchall()
        return Page(total, tuple(self._job_from_row(row) for row in rows), limit, offset)

    def claim_next(
        self,
        worker_id: str,
        *,
        lease_seconds: int = 300,
        now: Optional[str] = None,
    ) -> Optional[ResearchJob]:
        worker_id = _required_text(worker_id, "worker_id")
        if not isinstance(lease_seconds, int) or lease_seconds <= 0:
            raise ValueError("lease_seconds 必须是正整数")
        now_dt = _parse_or_now(now)
        now_value = now_dt.isoformat(timespec="microseconds")
        lease_expires = (now_dt + timedelta(seconds=lease_seconds)).isoformat(
            timespec="microseconds"
        )
        with self._transaction(immediate=True) as connection:
            candidate = connection.execute(
                """
                SELECT id, version FROM research_jobs
                WHERE status='queued'
                  AND (run_after IS NULL OR run_after<=?)
                  AND attempts < max_attempts
                ORDER BY priority DESC, COALESCE(run_after, ''), created_at, id
                LIMIT 1
                """,
                (now_value,),
            ).fetchone()
            if candidate is None:
                return None
            cursor = connection.execute(
                """
                UPDATE research_jobs
                SET status='running', attempts=attempts+1, lease_owner=?,
                    lease_expires_at=?, started_at=COALESCE(started_at, ?),
                    updated_at=?, version=version+1
                WHERE id=? AND status='queued' AND version=?
                """,
                (
                    worker_id,
                    lease_expires,
                    now_value,
                    now_value,
                    candidate["id"],
                    candidate["version"],
                ),
            )
            if not cursor.rowcount:
                raise JobStateConflictError(
                    f"Job {candidate['id']} claim compare-and-swap 失败"
                )
            row = connection.execute(
                "SELECT * FROM research_jobs WHERE id=?", (candidate["id"],)
            ).fetchone()
        return self._job_from_row(row)

    def transition(
        self,
        job_id: str,
        to_status: str,
        *,
        expected_status: str,
        expected_version: int,
        result: Any = None,
        error_code: Optional[str] = None,
        error_message: Optional[str] = None,
        lease_owner: Optional[str] = None,
    ) -> ResearchJob:
        _validate_status(to_status)
        _validate_status(expected_status)
        if to_status not in _ALLOWED_TRANSITIONS[expected_status]:
            raise ValueError(f"非法 job 状态转换：{expected_status} → {to_status}")
        result_json = None if result is None else _json_dump(result)
        now = _now_iso()
        terminal_at = now if to_status in _TERMINAL_STATUSES else None
        with self._transaction(immediate=True) as connection:
            cursor = connection.execute(
                """
                UPDATE research_jobs
                SET status=?, result=?, error_code=?, error_message=?,
                    lease_owner=CASE WHEN ? IN ('running', 'cancel_requested')
                                     THEN lease_owner ELSE NULL END,
                    lease_expires_at=CASE WHEN ? IN ('running', 'cancel_requested')
                                          THEN lease_expires_at ELSE NULL END,
                    finished_at=?, updated_at=?, version=version+1
                WHERE id=? AND status=? AND version=?
                  AND (? IS NULL OR lease_owner=?)
                """,
                (
                    to_status,
                    result_json,
                    error_code,
                    error_message,
                    to_status,
                    to_status,
                    terminal_at,
                    now,
                    job_id,
                    expected_status,
                    expected_version,
                    lease_owner,
                    lease_owner,
                ),
            )
            if not cursor.rowcount:
                self._raise_job_conflict_locked(
                    connection, job_id, expected_status, expected_version
                )
            row = connection.execute(
                "SELECT * FROM research_jobs WHERE id=?", (job_id,)
            ).fetchone()
        return self._job_from_row(row)

    def update_progress(
        self,
        job_id: str,
        current: int,
        *,
        total: Optional[int] = None,
        expected_version: int,
        expected_status: str = "running",
        lease_owner: Optional[str] = None,
    ) -> ResearchJob:
        if not isinstance(current, int) or current < 0:
            raise ValueError("current 必须是非负整数")
        if total is not None and (not isinstance(total, int) or total < current):
            raise ValueError("total 必须是不小于 current 的非负整数")
        _validate_status(expected_status)
        with self._transaction(immediate=True) as connection:
            cursor = connection.execute(
                """
                UPDATE research_jobs
                SET progress_current=?, progress_total=?, updated_at=?, version=version+1
                WHERE id=? AND status=? AND version=?
                  AND (? IS NULL OR lease_owner=?)
                """,
                (
                    current,
                    total,
                    _now_iso(),
                    job_id,
                    expected_status,
                    expected_version,
                    lease_owner,
                    lease_owner,
                ),
            )
            if not cursor.rowcount:
                self._raise_job_conflict_locked(
                    connection, job_id, expected_status, expected_version
                )
            row = connection.execute(
                "SELECT * FROM research_jobs WHERE id=?", (job_id,)
            ).fetchone()
        return self._job_from_row(row)

    def request_cancel(
        self,
        job_id: str,
        *,
        expected_version: int,
    ) -> ResearchJob:
        now = _now_iso()
        with self._transaction(immediate=True) as connection:
            row = connection.execute(
                "SELECT * FROM research_jobs WHERE id=?", (job_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"Research Job 不存在：{job_id}")
            if int(row["version"]) != expected_version:
                raise JobStateConflictError(
                    f"Research Job {job_id} 版本冲突：期望 {expected_version}，当前 {row['version']}"
                )
            if row["status"] in _TERMINAL_STATUSES or row["status"] == "cancel_requested":
                return self._job_from_row(row)
            if row["status"] not in {"queued", "running", "interrupted"}:
                raise JobStateConflictError(
                    f"Research Job {job_id} 当前状态不能请求取消：{row['status']}"
                )
            # 尚未运行或已经中断的任务没有 worker 会接手确认，直接进入终态；
            # 仅 running 使用 cancel_requested，由持有 lease 的 worker 在阶段边界取消。
            next_status = (
                "cancel_requested" if row["status"] == "running" else "cancelled"
            )
            cursor = connection.execute(
                """
                UPDATE research_jobs
                SET status=?, cancel_requested_at=?,
                    finished_at=CASE WHEN ?='cancelled' THEN ? ELSE finished_at END,
                    lease_owner=CASE WHEN ?='cancel_requested' THEN lease_owner ELSE NULL END,
                    lease_expires_at=CASE WHEN ?='cancel_requested' THEN lease_expires_at ELSE NULL END,
                    updated_at=?, version=version+1
                WHERE id=? AND version=?
                """,
                (
                    next_status,
                    now,
                    next_status,
                    now,
                    next_status,
                    next_status,
                    now,
                    job_id,
                    expected_version,
                ),
            )
            if not cursor.rowcount:
                raise JobStateConflictError(f"Research Job {job_id} 取消 CAS 失败")
            updated = connection.execute(
                "SELECT * FROM research_jobs WHERE id=?", (job_id,)
            ).fetchone()
        return self._job_from_row(updated)

    def interrupt_running(self) -> int:
        """进程启动恢复边界：遗留 running job 原子标记为 interrupted。"""

        now = _now_iso()
        with self._transaction(immediate=True) as connection:
            cursor = connection.execute(
                """
                UPDATE research_jobs
                SET status='interrupted', lease_owner=NULL, lease_expires_at=NULL,
                    updated_at=?, version=version+1
                WHERE status='running'
                """,
                (now,),
            )
        return int(cursor.rowcount)

    def recover_startup(self) -> tuple[int, int, int]:
        """恢复遗留任务，并且只重排仍有重试额度的幂等任务。

        返回 ``(interrupted, requeued, left_interrupted)``。整个恢复过程位于
        一个 ``BEGIN IMMEDIATE`` 事务，多个进程同时启动也不会重复重排。
        """

        now = _now_iso()
        with self._transaction(immediate=True) as connection:
            interrupted = int(
                connection.execute(
                    """
                    UPDATE research_jobs
                    SET status='interrupted', lease_owner=NULL, lease_expires_at=NULL,
                        error_code='worker_interrupted',
                        error_message='任务因服务重启而中断',
                        updated_at=?, version=version+1
                    WHERE status='running'
                    """,
                    (now,),
                ).rowcount
            )
            # 已请求取消的任务在旧 worker 消失后不应再次执行。
            connection.execute(
                """
                UPDATE research_jobs
                SET status='cancelled', lease_owner=NULL, lease_expires_at=NULL,
                    finished_at=?, updated_at=?, version=version+1
                WHERE status='cancel_requested'
                """,
                (now, now),
            )
            requeued = int(
                connection.execute(
                    """
                    UPDATE research_jobs
                    SET status='queued', progress_current=0, progress_total=NULL,
                        result=NULL, error_code=NULL, error_message=NULL,
                        lease_owner=NULL, lease_expires_at=NULL, finished_at=NULL,
                        run_after=NULL, updated_at=?, version=version+1
                    WHERE status='interrupted' AND idempotent=1
                      AND attempts < max_attempts
                    """,
                    (now,),
                ).rowcount
            )
            left = int(
                connection.execute(
                    "SELECT COUNT(*) FROM research_jobs WHERE status='interrupted'"
                ).fetchone()[0]
            )
        return interrupted, requeued, left

    def renew_lease(
        self,
        job_id: str,
        worker_id: str,
        *,
        expected_version: int,
        lease_seconds: int = 300,
        now: Optional[str] = None,
    ) -> ResearchJob:
        worker_id = _required_text(worker_id, "worker_id")
        if not isinstance(lease_seconds, int) or lease_seconds <= 0:
            raise ValueError("lease_seconds 必须是正整数")
        now_dt = _parse_or_now(now)
        lease_expires = (now_dt + timedelta(seconds=lease_seconds)).isoformat(
            timespec="microseconds"
        )
        with self._transaction(immediate=True) as connection:
            cursor = connection.execute(
                """
                UPDATE research_jobs
                SET lease_expires_at=?, updated_at=?, version=version+1
                WHERE id=? AND status IN ('running', 'cancel_requested')
                  AND lease_owner=? AND version=?
                """,
                (
                    lease_expires,
                    now_dt.isoformat(timespec="microseconds"),
                    job_id,
                    worker_id,
                    expected_version,
                ),
            )
            if not cursor.rowcount:
                self._raise_job_conflict_locked(
                    connection, job_id, "running", expected_version
                )
            row = connection.execute(
                "SELECT * FROM research_jobs WHERE id=?", (job_id,)
            ).fetchone()
        return self._job_from_row(row)

    def reap_expired_leases(self, *, now: Optional[str] = None) -> tuple[int, int]:
        """中断过期 lease，并重排仍有额度的幂等任务。"""

        now_value = _parse_or_now(now).isoformat(timespec="microseconds")
        with self._transaction(immediate=True) as connection:
            expired = int(
                connection.execute(
                    """
                    UPDATE research_jobs
                    SET status='interrupted', lease_owner=NULL, lease_expires_at=NULL,
                        error_code='lease_expired', error_message='任务执行租约已过期',
                        updated_at=?, version=version+1
                    WHERE status='running' AND lease_expires_at IS NOT NULL
                      AND lease_expires_at<=?
                    """,
                    (now_value, now_value),
                ).rowcount
            )
            requeued = int(
                connection.execute(
                    """
                    UPDATE research_jobs
                    SET status='queued', progress_current=0, progress_total=NULL,
                        error_code=NULL, error_message=NULL, run_after=NULL,
                        updated_at=?, version=version+1
                    WHERE status='interrupted' AND idempotent=1
                      AND attempts < max_attempts
                    """,
                    (now_value,),
                ).rowcount
            )
        return expired, requeued

    @staticmethod
    def _same_enqueue_request(
        row: sqlite3.Row,
        *,
        job_type: str,
        payload_json: str,
        project_id: Optional[str],
        artifact_id: Optional[str],
        priority: int,
        run_after: Optional[str],
        timeout_seconds: Optional[int],
        max_attempts: int,
        idempotent: bool,
    ) -> bool:
        return (
            row["job_type"] == job_type
            and row["payload"] == payload_json
            and row["project_id"] == project_id
            and row["artifact_id"] == artifact_id
            and int(row["priority"]) == priority
            and row["run_after"] == run_after
            and row["timeout_seconds"] == timeout_seconds
            and int(row["max_attempts"]) == max_attempts
            and bool(row["idempotent"]) == bool(idempotent)
        )

    @staticmethod
    def _validate_job_scope_locked(
        connection: sqlite3.Connection,
        *,
        project_id: Optional[str],
        artifact_id: Optional[str],
    ) -> None:
        if project_id is not None and connection.execute(
            "SELECT 1 FROM research_projects WHERE id=?", (project_id,)
        ).fetchone() is None:
            raise KeyError(f"研究项目不存在：{project_id}")
        if artifact_id is not None:
            artifact = connection.execute(
                "SELECT project_id FROM research_artifacts WHERE id=?", (artifact_id,)
            ).fetchone()
            if artifact is None:
                raise KeyError(f"研究 artifact 不存在：{artifact_id}")
            if project_id is not None and artifact["project_id"] != project_id:
                raise ValueError("job 的 project_id 与 artifact 所属项目不一致")

    @staticmethod
    def _translate_foreign_key_error_locked(
        connection: sqlite3.Connection,
        *,
        project_id: Optional[str],
        artifact_id: Optional[str],
    ) -> None:
        if project_id is not None and connection.execute(
            "SELECT 1 FROM research_projects WHERE id=?", (project_id,)
        ).fetchone() is None:
            raise KeyError(f"研究项目不存在：{project_id}")
        if artifact_id is not None and connection.execute(
            "SELECT 1 FROM research_artifacts WHERE id=?", (artifact_id,)
        ).fetchone() is None:
            raise KeyError(f"研究 artifact 不存在：{artifact_id}")

    @staticmethod
    def _raise_job_conflict_locked(
        connection: sqlite3.Connection,
        job_id: str,
        expected_status: str,
        expected_version: int,
    ) -> None:
        row = connection.execute(
            "SELECT status, version FROM research_jobs WHERE id=?", (job_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"Research Job 不存在：{job_id}")
        raise JobStateConflictError(
            f"Research Job {job_id} 状态/版本冲突：期望 "
            f"{expected_status}@{expected_version}，当前 {row['status']}@{row['version']}"
        )

    @staticmethod
    def _job_from_row(row: sqlite3.Row) -> ResearchJob:
        return ResearchJob(
            id=row["id"],
            project_id=row["project_id"],
            artifact_id=row["artifact_id"],
            job_type=row["job_type"],
            status=row["status"],
            payload=json.loads(row["payload"]),
            result=None if row["result"] is None else json.loads(row["result"]),
            error_code=row["error_code"],
            error_message=row["error_message"],
            progress_current=int(row["progress_current"]),
            progress_total=row["progress_total"],
            attempts=int(row["attempts"]),
            max_attempts=int(row["max_attempts"]),
            idempotent=bool(row["idempotent"]),
            priority=int(row["priority"]),
            run_after=row["run_after"],
            timeout_seconds=row["timeout_seconds"],
            idempotency_key=row["idempotency_key"],
            lease_owner=row["lease_owner"],
            lease_expires_at=row["lease_expires_at"],
            cancel_requested_at=row["cancel_requested_at"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
            version=int(row["version"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


def _required_text(value: Any, name: str) -> str:
    if value is None:
        raise ValueError(f"{name} 不能为空")
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} 不能为空")
    return text


def _validate_status(status: str) -> None:
    if status not in _JOB_STATUSES:
        raise ValueError(f"status 必须是以下值之一：{', '.join(sorted(_JOB_STATUSES))}")


def _validate_page(limit: int, offset: int) -> tuple[int, int]:
    if not isinstance(limit, int) or not 1 <= limit <= 200:
        raise ValueError("limit 必须是 1–200 的整数")
    if not isinstance(offset, int) or offset < 0:
        raise ValueError("offset 必须是非负整数")
    return limit, offset


def _json_dump(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("值必须可序列化为 JSON") from exc


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _parse_or_now(value: Optional[str]) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("now 必须是 ISO-8601 时间") from exc
    if parsed.tzinfo is None:
        raise ValueError("now 必须包含时区")
    return parsed.astimezone(timezone.utc)
