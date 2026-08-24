import pytest

from pragent.storage import (
    JobIdempotencyConflictError,
    JobRepository,
    JobStateConflictError,
    ResearchRepository,
)


def test_job_enqueue_idempotency_filtering_and_pagination(tmp_path):
    db_path = tmp_path / "jobs.db"
    research = ResearchRepository(db_path)
    project = research.create_project("任务项目")
    research.close()

    jobs = JobRepository(db_path)
    first = jobs.enqueue(
        "deep_read",
        {"source_id": "source_1"},
        project_id=project.id,
        priority=10,
        max_attempts=2,
        timeout_seconds=120,
        idempotent=True,
        idempotency_key="deep-read:source-1:v1",
    )
    same = jobs.enqueue(
        "deep_read",
        {"source_id": "source_1"},
        project_id=project.id,
        priority=10,
        max_attempts=2,
        timeout_seconds=120,
        idempotent=True,
        idempotency_key="deep-read:source-1:v1",
    )
    assert same.id == first.id
    with pytest.raises(JobIdempotencyConflictError, match="不同任务"):
        jobs.enqueue(
            "deep_read",
            {"source_id": "source_2"},
            project_id=project.id,
            idempotent=True,
            idempotency_key="deep-read:source-1:v1",
        )
    jobs.enqueue("export", {"format": "markdown"}, project_id=project.id)

    page = jobs.list(project_id=project.id, limit=1, offset=0)
    assert page.total == 2 and len(page.items) == 1
    assert jobs.list(job_type="deep_read").items == (first,)
    with pytest.raises(ValueError, match="idempotency_key"):
        jobs.enqueue("deep_read", {}, idempotent=True)
    with pytest.raises(KeyError, match="研究项目不存在"):
        jobs.enqueue("deep_read", {}, project_id="project_missing")
    jobs.close()


def test_job_claim_progress_transition_and_cross_connection_cas(tmp_path):
    db_path = tmp_path / "claim.db"
    first = JobRepository(db_path)
    second = JobRepository(db_path)
    queued = first.enqueue("deep_read", {"source_id": "s"}, priority=5)

    claimed = first.claim_next(
        "worker-a", now="2026-01-01T00:00:00+00:00", lease_seconds=30
    )
    assert claimed.id == queued.id
    assert claimed.status == "running"
    assert claimed.attempts == 1
    assert claimed.lease_owner == "worker-a"
    assert second.claim_next(
        "worker-b", now="2026-01-01T00:00:01+00:00"
    ) is None

    progressed = second.update_progress(
        claimed.id,
        2,
        total=5,
        expected_version=claimed.version,
    )
    assert (progressed.progress_current, progressed.progress_total) == (2, 5)
    with pytest.raises(JobStateConflictError, match="状态/版本冲突"):
        first.update_progress(
            claimed.id,
            3,
            total=5,
            expected_version=claimed.version,
        )

    succeeded = first.transition(
        claimed.id,
        "succeeded",
        expected_status="running",
        expected_version=progressed.version,
        result={"artifact_id": "artifact_1"},
    )
    assert succeeded.status == "succeeded"
    assert succeeded.result == {"artifact_id": "artifact_1"}
    assert succeeded.finished_at is not None
    assert succeeded.lease_owner is None
    with pytest.raises(ValueError, match="非法"):
        first.transition(
            succeeded.id,
            "running",
            expected_status="succeeded",
            expected_version=succeeded.version,
        )
    first.close()
    second.close()


def test_job_cancel_interrupt_and_reopen(tmp_path):
    db_path = tmp_path / "recovery.db"
    jobs = JobRepository(db_path)
    cancel_job = jobs.enqueue("web_fetch", {"url": "https://example.org"})
    cancelled = jobs.request_cancel(
        cancel_job.id, expected_version=cancel_job.version
    )
    assert cancelled.status == "cancelled"
    assert cancelled.finished_at is not None

    running_one = jobs.enqueue("deep_read", {"n": 1})
    running_two = jobs.enqueue("deep_read", {"n": 2})
    first_claim = jobs.claim_next("worker")
    second_claim = jobs.claim_next("worker")
    assert {first_claim.id, second_claim.id} == {running_one.id, running_two.id}
    assert jobs.interrupt_running() == 2
    assert jobs.get(first_claim.id).status == "interrupted"
    jobs.close()

    reopened = JobRepository(db_path)
    assert reopened.get(cancelled.id).status == "cancelled"
    assert reopened.list(status="interrupted").total == 2
    interrupted = reopened.get(first_claim.id)
    requeued = reopened.transition(
        interrupted.id,
        "queued",
        expected_status="interrupted",
        expected_version=interrupted.version,
    )
    assert requeued.status == "queued"
    reopened.close()
