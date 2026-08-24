"""固定并发、协作取消的持久任务 worker。"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

from pragent.models import ResearchJob
from pragent.storage import JobStateConflictError

from .queue import JobQueue


class JobCancelled(RuntimeError):
    pass


class JobDeadlineExceeded(RuntimeError):
    pass


@dataclass
class JobContext:
    queue: JobQueue
    worker_id: str
    job: ResearchJob
    deadline: Optional[float]
    stop_event: threading.Event

    @property
    def remaining_seconds(self) -> Optional[float]:
        if self.deadline is None:
            return None
        return max(0.0, self.deadline - time.monotonic())

    def check_cancelled(self) -> None:
        current = self.queue.repository.get(self.job.id)
        if current is None:
            raise JobCancelled("任务已不存在")
        self.job = current
        if current.status == "cancel_requested" or self.stop_event.is_set():
            raise JobCancelled("任务已取消")
        if current.status != "running" or current.lease_owner != self.worker_id:
            raise JobCancelled("任务执行租约已丢失")
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise JobDeadlineExceeded("任务超过执行时限")

    def report_progress(self, current: int, total: Optional[int] = None) -> None:
        self.check_cancelled()
        self.job = self.queue.repository.update_progress(
            self.job.id,
            current,
            total=total,
            expected_version=self.job.version,
            lease_owner=self.worker_id,
        )

    def renew_lease(self, lease_seconds: int) -> None:
        self.check_cancelled()
        self.job = self.queue.repository.renew_lease(
            self.job.id,
            self.worker_id,
            expected_version=self.job.version,
            lease_seconds=lease_seconds,
        )


JobHandler = Callable[[JobContext, Any], Any]


class WorkerPool:
    """SQLite claim + 固定线程数的有界执行器。

    timeout 和取消只在 handler 调用 ``JobContext`` 的阶段边界生效；不会用
    future 超时伪装已经停止一个仍可能产生副作用的线程。
    """

    def __init__(
        self,
        queue: JobQueue,
        handlers: Mapping[str, JobHandler],
        *,
        worker_count: int = 2,
        poll_interval: float = 0.25,
        lease_seconds: int = 60,
        worker_prefix: Optional[str] = None,
    ) -> None:
        if not isinstance(worker_count, int) or not 1 <= worker_count <= 16:
            raise ValueError("worker_count 必须是 1–16 的整数")
        if poll_interval <= 0:
            raise ValueError("poll_interval 必须大于 0")
        if not isinstance(lease_seconds, int) or lease_seconds <= 0:
            raise ValueError("lease_seconds 必须是正整数")
        self.queue = queue
        self.handlers = dict(handlers)
        self.worker_count = worker_count
        self.poll_interval = float(poll_interval)
        self.lease_seconds = lease_seconds
        self.worker_prefix = worker_prefix or f"worker-{uuid.uuid4().hex[:10]}"
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._lock = threading.Lock()

    @property
    def running(self) -> bool:
        return any(thread.is_alive() for thread in self._threads)

    def start(self) -> None:
        with self._lock:
            if self.running:
                return
            self._stop.clear()
            self._threads = [
                threading.Thread(
                    target=self._run_loop,
                    args=(f"{self.worker_prefix}-{index + 1}",),
                    name=f"pra-job-{index + 1}",
                    daemon=True,
                )
                for index in range(self.worker_count)
            ]
            for thread in self._threads:
                thread.start()

    def stop(self, *, grace_seconds: float = 10.0) -> None:
        if grace_seconds < 0:
            raise ValueError("grace_seconds 不能为负数")
        self._stop.set()
        deadline = time.monotonic() + grace_seconds
        for thread in tuple(self._threads):
            thread.join(max(0.0, deadline - time.monotonic()))

    def run_once(self, worker_id: str = "worker-once") -> bool:
        """同步处理至多一个任务，供测试和显式单步 worker 使用。"""

        self.queue.reap_expired()
        job = self.queue.claim(worker_id, lease_seconds=self.lease_seconds)
        if job is None:
            return False
        self._execute(worker_id, job)
        return True

    def _run_loop(self, worker_id: str) -> None:
        while not self._stop.is_set():
            try:
                did_work = self.run_once(worker_id)
            except Exception:
                # 单个 repository/handler 故障不能终止整个固定 worker。
                did_work = False
            if not did_work:
                self._stop.wait(self.poll_interval)

    def _execute(self, worker_id: str, job: ResearchJob) -> None:
        deadline = (
            None
            if job.timeout_seconds is None
            else time.monotonic() + job.timeout_seconds
        )
        context = JobContext(self.queue, worker_id, job, deadline, self._stop)
        handler = self.handlers.get(job.job_type)
        if handler is None:
            self._finish_error(
                context,
                "unknown_job_type",
                "任务类型没有可用处理器",
            )
            return
        try:
            context.check_cancelled()
            result = handler(context, job.payload)
            context.check_cancelled()
        except JobCancelled:
            self._finish_cancelled(context)
            return
        except JobDeadlineExceeded:
            self._finish_error(context, "deadline_exceeded", "任务超过执行时限")
            return
        except Exception:
            self._finish_error(
                context,
                "handler_failed",
                "任务执行失败，请检查本地日志",
            )
            return
        try:
            context.job = self.queue.repository.transition(
                context.job.id,
                "succeeded",
                expected_status="running",
                expected_version=context.job.version,
                result=result,
                lease_owner=worker_id,
            )
        except JobStateConflictError:
            # cancel/lease race 由持有最新版本的一方决定，旧 worker 不覆盖。
            return

    def _finish_cancelled(self, context: JobContext) -> None:
        current = self.queue.repository.get(context.job.id)
        if current is None:
            return
        if current.status == "running" and self._stop.is_set():
            target = "interrupted"
        elif current.status == "cancel_requested":
            target = "cancelled"
        else:
            return
        try:
            self.queue.repository.transition(
                current.id,
                target,
                expected_status=current.status,
                expected_version=current.version,
                error_code="worker_stopped" if target == "interrupted" else None,
                error_message="任务因 worker 停止而中断" if target == "interrupted" else None,
                lease_owner=context.worker_id,
            )
        except JobStateConflictError:
            return

    def _finish_error(self, context: JobContext, code: str, message: str) -> None:
        current = self.queue.repository.get(context.job.id)
        if current is None or current.status not in {"running", "cancel_requested"}:
            return
        try:
            self.queue.repository.transition(
                current.id,
                "failed",
                expected_status=current.status,
                expected_version=current.version,
                error_code=code,
                error_message=message,
                lease_owner=context.worker_id,
            )
        except JobStateConflictError:
            return
