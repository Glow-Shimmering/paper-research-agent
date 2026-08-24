import threading
import time

import pytest

from pragent.jobs import JobQueue, WorkerPool
from pragent.storage import JobRepository, JobStateConflictError


def test_startup_recovery_only_requeues_idempotent_jobs_with_attempt_budget(tmp_path):
    repository = JobRepository(tmp_path / "jobs.db")
    queue = JobQueue(repository)
    retryable = repository.enqueue(
        "deep_read",
        {"source_id": "s1"},
        idempotent=True,
        idempotency_key="deep:s1",
        max_attempts=2,
    )
    non_idempotent = repository.enqueue("export", {}, max_attempts=2)
    exhausted = repository.enqueue(
        "deep_read",
        {"source_id": "s2"},
        idempotent=True,
        idempotency_key="deep:s2",
        max_attempts=1,
    )
    for _ in range(3):
        assert repository.claim_next("old-worker") is not None

    report = queue.recover_startup()

    assert report.interrupted == 3
    assert report.requeued == 1
    assert report.left_interrupted == 2
    assert repository.get(retryable.id).status == "queued"
    assert repository.get(non_idempotent.id).status == "interrupted"
    assert repository.get(exhausted.id).status == "interrupted"
    repository.close()


def test_worker_dispatches_progress_and_enforces_lease_owner(tmp_path):
    repository = JobRepository(tmp_path / "jobs.db")
    queue = JobQueue(repository)
    queued = queue.enqueue("sum", {"values": [2, 3]}, timeout_seconds=5)

    def handler(context, payload):
        context.report_progress(1, 2)
        context.renew_lease(30)
        context.report_progress(2, 2)
        return {"value": sum(payload["values"])}

    worker = WorkerPool(queue, {"sum": handler}, worker_count=1)
    assert worker.run_once("owner") is True
    completed = repository.get(queued.id)
    assert completed.status == "succeeded"
    assert completed.result == {"value": 5}
    assert (completed.progress_current, completed.progress_total) == (2, 2)

    another = queue.enqueue("sum", {"values": []})
    claimed = repository.claim_next("right-owner")
    with pytest.raises(JobStateConflictError):
        repository.transition(
            claimed.id,
            "succeeded",
            expected_status="running",
            expected_version=claimed.version,
            lease_owner="wrong-owner",
        )
    repository.close()


def test_running_cancel_is_cooperative_and_queued_cancel_is_immediate(tmp_path):
    repository = JobRepository(tmp_path / "jobs.db")
    queue = JobQueue(repository)
    queued = queue.enqueue("wait", {})
    cancelled = queue.cancel(queued.id, expected_version=queued.version)
    assert cancelled.status == "cancelled"

    running = queue.enqueue("wait", {}, timeout_seconds=5)
    entered = threading.Event()
    proceed = threading.Event()

    def handler(context, _payload):
        entered.set()
        assert proceed.wait(2)
        context.check_cancelled()

    worker = WorkerPool(queue, {"wait": handler}, worker_count=1)
    thread = threading.Thread(target=worker.run_once, args=("cancel-owner",))
    thread.start()
    assert entered.wait(2)
    current = repository.get(running.id)
    requested = queue.cancel(current.id, expected_version=current.version)
    assert requested.status == "cancel_requested"
    proceed.set()
    thread.join(3)
    assert repository.get(running.id).status == "cancelled"
    repository.close()


def test_worker_pool_has_fixed_bounded_concurrency_and_survives_failures(tmp_path):
    repository = JobRepository(tmp_path / "jobs.db")
    queue = JobQueue(repository)
    ids = [queue.enqueue("bounded", {"n": n}).id for n in range(5)]
    bad = queue.enqueue("missing", {}).id
    lock = threading.Lock()
    active = 0
    maximum = 0

    def handler(_context, payload):
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        time.sleep(0.03)
        with lock:
            active -= 1
        return payload

    pool = WorkerPool(
        queue,
        {"bounded": handler},
        worker_count=2,
        poll_interval=0.01,
    )
    pool.start()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if all(repository.get(job_id).status in {"succeeded", "failed"} for job_id in [*ids, bad]):
            break
        time.sleep(0.01)
    pool.stop(grace_seconds=2)

    assert 1 <= maximum <= 2
    assert all(repository.get(job_id).status == "succeeded" for job_id in ids)
    missing = repository.get(bad)
    assert missing.status == "failed"
    assert missing.error_code == "unknown_job_type"
    assert not pool.running
    repository.close()
