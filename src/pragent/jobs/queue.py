"""持久 research job 队列的意图级门面。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from pragent.models import ResearchJob
from pragent.storage import JobRepository


@dataclass(frozen=True)
class RecoveryReport:
    interrupted: int
    requeued: int
    left_interrupted: int


class JobQueue:
    """把队列语义集中到 repository CAS 之上，不复制持久化状态。"""

    def __init__(self, repository: JobRepository) -> None:
        self.repository = repository

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
    ) -> ResearchJob:
        return self.repository.enqueue(
            job_type,
            payload,
            project_id=project_id,
            artifact_id=artifact_id,
            priority=priority,
            run_after=run_after,
            timeout_seconds=timeout_seconds,
            max_attempts=max_attempts,
            idempotent=idempotent,
            idempotency_key=idempotency_key,
        )

    def claim(self, worker_id: str, *, lease_seconds: int) -> Optional[ResearchJob]:
        return self.repository.claim_next(worker_id, lease_seconds=lease_seconds)

    def cancel(self, job_id: str, *, expected_version: int) -> ResearchJob:
        return self.repository.request_cancel(job_id, expected_version=expected_version)

    def recover_startup(self) -> RecoveryReport:
        interrupted, requeued, left = self.repository.recover_startup()
        return RecoveryReport(interrupted, requeued, left)

    def reap_expired(self) -> tuple[int, int]:
        return self.repository.reap_expired_leases()
