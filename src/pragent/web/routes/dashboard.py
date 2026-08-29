"""Dashboard、任务中心与中文帮助页面（JSON API + HTMX UI）。

Dashboard 聚合项目、最近来源与进行中任务；任务中心提供全局 job
轮询、筛选与取消请求。所有响应只返回安全字段，不暴露主机路径或
payload 内部细节。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from pragent.storage import JobStateConflictError

from .projects import _is_htmx, _render, _render_error, _validated_form

_JOB_TYPE_LABELS = {
    "deep_read": "单篇精读",
    "deep_read_field": "精读字段重生成",
    "comparison": "跨论文比较",
    "review_outline": "综述提纲",
    "review_section": "综述章节",
    "export": "导出",
}
_ACTIVE_STATUSES = ("queued", "running", "cancel_requested")
_JOB_STATUSES = (
    "queued",
    "running",
    "succeeded",
    "failed",
    "cancel_requested",
    "cancelled",
    "interrupted",
)
_MAX_EVIDENCE_ITEMS = 100


def register_dashboard_routes(
    app: FastAPI,
    *,
    repository_factory: Callable,
    job_queue_factory: Callable,
    store_factory: Callable,
    templates_directory: str,
) -> None:
    templates = Jinja2Templates(directory=templates_directory)

    # ---------- JSON API ----------

    @app.get("/api/v1/jobs")
    def api_jobs(
        status: Optional[str] = Query(None),
        job_type: Optional[str] = Query(None, max_length=64),
        project_id: Optional[str] = Query(None, max_length=128),
        limit: int = Query(50, ge=1, le=200),
        offset: int = Query(0, ge=0),
    ):
        try:
            page = job_queue_factory().repository.list(
                status=status, job_type=job_type, project_id=project_id,
                limit=limit, offset=offset,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "total": page.total,
            "limit": page.limit,
            "offset": page.offset,
            "items": [_job_dict(job) for job in page.items],
        }

    @app.get("/api/v1/jobs/{job_id}")
    def api_job(job_id: str):
        job = job_queue_factory().repository.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="任务不存在")
        return _job_dict(job)

    @app.post("/api/v1/jobs/{job_id}/cancellation", status_code=202)
    def api_cancel_job(job_id: str, payload: dict):
        expected_version = _required_int(payload, "expected_version")
        try:
            job = job_queue_factory().cancel(job_id, expected_version=expected_version)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="任务不存在") from exc
        except JobStateConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return _job_dict(job)

    @app.get("/api/v1/projects/{project_id}/evidence")
    def api_project_evidence(project_id: str):
        repository = repository_factory()
        try:
            _require_project_exists(repository, project_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"items": _project_evidence_items(repository, store_factory(), project_id)}

    # ---------- server-rendered HTMX UI ----------

    @app.get("/ui/", response_class=HTMLResponse)
    @app.get("/ui/dashboard", response_class=HTMLResponse)
    def ui_dashboard(request: Request):
        repository = repository_factory()
        queue = job_queue_factory().repository
        projects_page = repository.list_projects(limit=6, offset=0)
        sources_page = repository.list_sources(limit=6, offset=0)
        active_jobs: list[Any] = []
        for status in _ACTIVE_STATUSES:
            active_jobs.extend(queue.list(status=status, limit=20, offset=0).items)
        active_jobs.sort(key=lambda job: job.created_at, reverse=True)
        papers, chunks = store_factory().stats()
        return _render(
            templates,
            request,
            "dashboard.html",
            {
                "projects": projects_page.items,
                "project_total": projects_page.total,
                "sources": sources_page.items,
                "source_total": sources_page.total,
                "active_jobs": active_jobs,
                "paper_count": papers,
                "chunk_count": chunks,
                "type_labels": _JOB_TYPE_LABELS,
            },
        )

    @app.get("/ui/jobs", response_class=HTMLResponse)
    def ui_jobs(request: Request, status: Optional[str] = Query(None)):
        if status is not None and status not in _JOB_STATUSES:
            return _render_error(
                templates, request, f"未知任务状态筛选：{status}", status_code=400
            )
        return _render(
            templates,
            request,
            "jobs.html",
            _jobs_context(job_queue_factory().repository, status),
        )

    @app.get("/ui/jobs/fragment", response_class=HTMLResponse)
    def ui_jobs_fragment(request: Request, status: Optional[str] = Query(None)):
        if status is not None and status not in _JOB_STATUSES:
            return _render_error(
                templates, request, f"未知任务状态筛选：{status}", status_code=400
            )
        return _render(
            templates,
            request,
            "fragments/job_list.html",
            _jobs_context(job_queue_factory().repository, status),
        )

    @app.post("/ui/jobs/{job_id}/cancel")
    async def ui_cancel_job(request: Request, job_id: str):
        form = await _validated_form(request)
        queue = job_queue_factory()
        try:
            expected_version = int(form.get("expected_version", ""))
        except ValueError as exc:
            return _render_error(templates, request, "expected_version 必须是整数", status_code=400)
        try:
            queue.cancel(job_id, expected_version=expected_version)
        except KeyError:
            return _render_error(templates, request, "任务不存在", status_code=404)
        except JobStateConflictError as exc:
            return _render_error(templates, request, str(exc), status_code=409)
        if not _is_htmx(request):
            return RedirectResponse("/ui/jobs", status_code=303)
        return _render(
            templates,
            request,
            "fragments/job_list.html",
            _jobs_context(queue.repository, None),
        )

    @app.get("/ui/help", response_class=HTMLResponse)
    def ui_help(request: Request):
        return _render(templates, request, "help.html", {})

    @app.get("/ui/projects/{project_id}/evidence", response_class=HTMLResponse)
    def ui_project_evidence(request: Request, project_id: str):
        repository = repository_factory()
        try:
            _require_project_exists(repository, project_id)
        except KeyError as exc:
            return _render_error(
                templates, request, str(exc), status_code=404
            )
        memberships = repository.list_project_sources(
            project_id, limit=200, offset=0
        ).items
        return _render(
            templates,
            request,
            "evidence_notes.html",
            {
                "project": repository.get_project(project_id),
                "evidence_items": _project_evidence_items(
                    repository, store_factory(), project_id
                ),
                "memberships": memberships,
                "notes": repository.list_notes(project_id, limit=100, offset=0).items,
            },
        )


def _jobs_context(repository, status: Optional[str]) -> dict[str, Any]:
    page = repository.list(status=status, limit=50, offset=0)
    return {
        "jobs": page.items,
        "job_total": page.total,
        "status_filter": status,
        "job_statuses": _JOB_STATUSES,
        "type_labels": _JOB_TYPE_LABELS,
    }


def _job_dict(job) -> dict[str, Any]:
    return {
        "id": job.id,
        "job_type": job.job_type,
        "type_label": _JOB_TYPE_LABELS.get(job.job_type, job.job_type),
        "status": job.status,
        "project_id": job.project_id,
        "artifact_id": job.artifact_id,
        "progress_current": job.progress_current,
        "progress_total": job.progress_total,
        "error_code": job.error_code,
        "error_message": job.error_message,
        "attempts": job.attempts,
        "max_attempts": job.max_attempts,
        "idempotent": job.idempotent,
        "cancel_requested": job.status == "cancel_requested"
        or job.cancel_requested_at is not None,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "finished_at": job.finished_at,
        "version": job.version,
    }


def _project_evidence_items(repository, store, project_id: str) -> list[dict[str, Any]]:
    """聚合项目内 artifact 当前 revision 引用的 evidence（复用工具口径）。"""
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for artifact in repository.list_artifacts(project_id, limit=200).items:
        revision = repository.get_current_artifact_revision(artifact.id)
        if revision is None:
            continue
        for link in repository.list_artifact_evidence(revision.id):
            if link.evidence_id in seen or len(items) >= _MAX_EVIDENCE_ITEMS:
                continue
            evidence = store.get_evidence(link.evidence_id)
            if evidence is None:
                continue
            seen.add(link.evidence_id)
            items.append(
                {
                    "evidence_id": link.evidence_id,
                    "field_path": link.field_path,
                    "artifact_id": artifact.id,
                    "artifact_type": artifact.artifact_type,
                    "title": getattr(evidence, "title", None),
                    "page": getattr(evidence, "page", None),
                    "stale": bool(getattr(evidence, "stale", False)),
                    "stale_reason": getattr(evidence, "stale_reason", None),
                    "source_kind": str(
                        getattr(evidence, "source_kind", "pdf") or "pdf"
                    ),
                    "locator": _safe_evidence_locator(evidence),
                    "preview": str(getattr(evidence, "text", "") or "")[:300],
                }
            )
    return items


def _safe_evidence_locator(evidence) -> Optional[str]:
    """返回不含主机路径的来源定位：网页用 canonical URI，PDF 用文件名。"""
    if str(getattr(evidence, "source_kind", "pdf") or "pdf") == "web":
        uri = getattr(evidence, "canonical_uri", None)
        return str(uri) if uri else None
    path = getattr(evidence, "path", None)
    return Path(str(path)).name if path else None


def _require_project_exists(repository, project_id: str) -> None:
    if repository.get_project(project_id) is None:
        raise KeyError(f"研究项目不存在：{project_id}")


def _required_int(payload: dict, name: str) -> int:
    value = (payload or {}).get(name)
    if isinstance(value, bool):
        raise HTTPException(status_code=400, detail=f"{name} 必须是整数")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"{name} 必须是整数") from exc
    if parsed < 1:
        raise HTTPException(status_code=400, detail=f"{name} 必须是正整数")
    return parsed
