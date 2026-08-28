"""Web preview, persistent export jobs, and scoped file downloads."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates

from pragent.exporting import ArtifactExportService, ExportError

from .projects import (
    _exception_message,
    _render,
    _render_error,
    _require_project,
    _validated_form,
)

_FORMATS = {"markdown", "docx", "csv", "json"}
_SAFE_JOB_ERRORS = {
    "unknown_job_type": "导出处理器不可用",
    "deadline_exceeded": "导出超过执行时限",
    "handler_failed": "导出失败，请刷新后重试",
    "worker_interrupted": "导出因服务重启而中断",
    "lease_expired": "导出任务执行租约已过期",
}


def register_export_routes(
    app: FastAPI,
    *,
    repository_factory: Callable,
    job_queue_factory: Callable,
    store_factory: Callable,
    export_directory_factory: Callable,
    templates_directory: str,
) -> None:
    templates = Jinja2Templates(directory=templates_directory)

    def service() -> ArtifactExportService:
        return ArtifactExportService(repository_factory(), store_factory())

    @app.get(
        "/api/v1/projects/{project_id}/artifacts/{artifact_id}/exports/preview"
    )
    def api_export_preview(project_id: str, artifact_id: str):
        try:
            snapshot = _snapshot(service(), project_id, artifact_id)
            preview, truncated = _markdown_preview(service(), snapshot)
            return {
                "artifact_id": artifact_id,
                "revision_id": snapshot.revision.id,
                "format": "markdown",
                "truncated": truncated,
                "content": preview,
            }
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=_exception_message(exc)) from exc
        except ExportError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post(
        "/api/v1/projects/{project_id}/artifacts/{artifact_id}/exports",
        status_code=202,
    )
    def api_start_export(project_id: str, artifact_id: str, payload: dict):
        try:
            result = _enqueue_export(
                service(),
                job_queue_factory(),
                project_id,
                artifact_id,
                str((payload or {}).get("format") or ""),
            )
            return result
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=_exception_message(exc)) from exc
        except (ExportError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get(
        "/api/v1/projects/{project_id}/artifacts/{artifact_id}/exports/jobs/{job_id}"
    )
    def api_export_job(project_id: str, artifact_id: str, job_id: str):
        job = _scoped_export_job(
            job_queue_factory(), project_id, artifact_id, job_id
        )
        return _public_job(job)

    @app.get(
        "/api/v1/projects/{project_id}/artifacts/{artifact_id}/exports/jobs/"
        "{job_id}/files/{filename}"
    )
    def api_download_export(
        project_id: str, artifact_id: str, job_id: str, filename: str
    ):
        job = _scoped_export_job(
            job_queue_factory(), project_id, artifact_id, job_id
        )
        path, media_type = _download_file(
            export_directory_factory(), job, filename
        )
        return FileResponse(path, media_type=media_type, filename=filename)

    @app.get(
        "/ui/projects/{project_id}/artifacts/{artifact_id}/exports",
        response_class=HTMLResponse,
    )
    def ui_exports(request: Request, project_id: str, artifact_id: str):
        try:
            snapshot = _snapshot(service(), project_id, artifact_id)
            preview, truncated = _markdown_preview(service(), snapshot)
            return _render(
                templates,
                request,
                "exports.html",
                {
                    "project": snapshot.project,
                    "artifact": snapshot.artifact,
                    "revision": snapshot.revision,
                    "freshness": snapshot.freshness,
                    "preview": preview,
                    "truncated": truncated,
                },
            )
        except KeyError:
            return _render_error(
                templates, request, "导出对象不存在", status_code=404
            )
        except ExportError as exc:
            return _render_error(
                templates, request, str(exc), status_code=409
            )

    @app.get(
        "/ui/projects/{project_id}/artifacts/{artifact_id}/exports/preview",
        response_class=HTMLResponse,
    )
    def ui_export_preview(request: Request, project_id: str, artifact_id: str):
        try:
            snapshot = _snapshot(service(), project_id, artifact_id)
            preview, truncated = _markdown_preview(service(), snapshot)
            return _render(
                templates,
                request,
                "fragments/export_preview.html",
                {
                    "revision": snapshot.revision,
                    "preview": preview,
                    "truncated": truncated,
                },
            )
        except (KeyError, ExportError) as exc:
            return _render_error(
                templates, request, _exception_message(exc), status_code=409
            )

    @app.post(
        "/ui/projects/{project_id}/artifacts/{artifact_id}/exports",
        response_class=HTMLResponse,
    )
    async def ui_start_export(
        request: Request, project_id: str, artifact_id: str
    ):
        form = await _validated_form(request)
        try:
            result = _enqueue_export(
                service(),
                job_queue_factory(),
                project_id,
                artifact_id,
                form.get("format", ""),
            )
            return _render(
                templates,
                request,
                "fragments/export_action.html",
                {"project_id": project_id, "artifact_id": artifact_id, **result},
            )
        except (KeyError, ExportError, ValueError) as exc:
            return _render_error(
                templates, request, _exception_message(exc), status_code=400
            )

    @app.get(
        "/ui/projects/{project_id}/artifacts/{artifact_id}/exports/jobs/{job_id}",
        response_class=HTMLResponse,
    )
    def ui_export_job(
        request: Request, project_id: str, artifact_id: str, job_id: str
    ):
        try:
            job = _scoped_export_job(
                job_queue_factory(), project_id, artifact_id, job_id
            )
            return _render(
                templates,
                request,
                "fragments/export_action.html",
                {
                    "project_id": project_id,
                    "artifact_id": artifact_id,
                    "job": _public_job(job),
                },
            )
        except KeyError:
            return _render_error(
                templates, request, "导出任务不存在", status_code=404
            )


def _snapshot(service, project_id, artifact_id):
    snapshot = service.freeze_current(artifact_id)
    if snapshot.project.id != project_id:
        raise KeyError("artifact 不存在")
    return snapshot


def _markdown_preview(service, snapshot, limit: int = 200_000):
    content = service.render(snapshot, "markdown")[0].data.decode("utf-8")
    return content[:limit], len(content) > limit


def _enqueue_export(service, queue, project_id, artifact_id, format):
    normalized = str(format).strip().lower()
    if normalized not in _FORMATS:
        raise ValueError("导出格式必须是 markdown、docx、csv 或 json")
    snapshot = _snapshot(service, project_id, artifact_id)
    payload = {
        "project_id": project_id,
        "artifact_id": artifact_id,
        "expected_revision_id": snapshot.revision.id,
        "citation_style": snapshot.citation_style,
        "source_versions": {
            item.source.id: item.source.version for item in snapshot.sources
        },
        "freshness_fingerprint": snapshot.freshness.current_fingerprint,
        "review_section_revisions": {
            item.artifact.id: item.revision.id for item in snapshot.review_sections
        },
        "format": normalized,
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    job = queue.enqueue(
        "export",
        payload,
        project_id=project_id,
        artifact_id=artifact_id,
        timeout_seconds=180,
        max_attempts=2,
        idempotent=True,
        idempotency_key="export:" + hashlib.sha256(canonical.encode()).hexdigest(),
    )
    return {"job": _public_job(job)}


def _scoped_export_job(queue, project_id, artifact_id, job_id):
    job = queue.repository.get(job_id)
    if (
        job is None
        or job.job_type != "export"
        or job.project_id != project_id
        or job.artifact_id != artifact_id
    ):
        raise HTTPException(status_code=404, detail="导出任务不存在")
    return job


def _download_file(export_directory, job, filename):
    result = job.result if job.status == "succeeded" else None
    files = result.get("files", []) if isinstance(result, dict) else []
    selected = next(
        (
            item
            for item in files
            if isinstance(item, dict) and item.get("name") == filename
        ),
        None,
    )
    if selected is None or Path(filename).name != filename:
        raise HTTPException(status_code=404, detail="导出文件不存在")
    root = (Path(export_directory) / job.id).resolve()
    path = (root / filename).resolve()
    if path.parent != root or not path.is_file():
        raise HTTPException(status_code=404, detail="导出文件不存在")
    return path, str(selected.get("media_type") or "application/octet-stream")


def _public_job(job):
    error = None
    if job.error_code:
        error = {
            "code": job.error_code,
            "message": _SAFE_JOB_ERRORS.get(job.error_code, "导出失败"),
        }
    result = job.result if isinstance(job.result, dict) else {}
    return {
        "id": job.id,
        "status": job.status,
        "progress_current": job.progress_current,
        "progress_total": job.progress_total,
        "error": error,
        "terminal": job.status in {"succeeded", "failed", "cancelled", "interrupted"},
        "revision_id": result.get("revision_id"),
        "format": result.get("format"),
        "files": result.get("files", []),
    }
