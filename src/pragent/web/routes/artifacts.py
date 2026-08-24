"""Deep Read JSON API 与 HTMX 页面。"""

from __future__ import annotations

from typing import Any, Callable, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from pragent.research import (
    DEEP_READ_FIELD_LABELS,
    DEEP_READ_FIELD_ORDER,
    DeepReadArtifactService,
    DeepReadCard,
)
from pragent.storage import RecordVersionConflictError

from .projects import (
    _exception_message,
    _form_int,
    _render,
    _render_error,
    _require_project,
    _validated_form,
)

_SAFE_JOB_ERRORS = {
    "unknown_job_type": "任务处理器不可用",
    "deadline_exceeded": "任务超过执行时限",
    "handler_failed": "任务执行失败，请检查本地日志",
    "worker_interrupted": "任务因服务重启而中断",
    "lease_expired": "任务执行租约已过期",
}


def register_artifact_routes(
    app: FastAPI,
    *,
    repository_factory: Callable,
    job_queue_factory: Callable,
    store_factory: Callable,
    templates_directory: str,
) -> None:
    templates = Jinja2Templates(directory=templates_directory)

    @app.get("/api/v1/projects/{project_id}/deep-reads")
    def api_list_deep_reads(project_id: str, limit: int = Query(50, ge=1, le=200)):
        repository = repository_factory()
        _require_project(repository, project_id)
        page = repository.list_artifacts(
            project_id, artifact_type="deep_read", limit=limit
        )
        return {"items": [_artifact_summary(repository, item) for item in page.items]}

    @app.post(
        "/api/v1/projects/{project_id}/sources/{source_id}/deep-reads",
        status_code=202,
    )
    def api_generate_deep_read(project_id: str, source_id: str):
        try:
            artifact, job = _enqueue_full(
                repository_factory(), job_queue_factory(), project_id, source_id
            )
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=_exception_message(exc)) from exc
        return {
            "artifact": _artifact_summary(repository_factory(), artifact),
            "job": _public_job(job),
        }

    @app.get("/api/v1/projects/{project_id}/deep-reads/{artifact_id}")
    def api_get_deep_read(
        project_id: str,
        artifact_id: str,
        revision_id: Optional[str] = None,
    ):
        try:
            return _detail_context(
                repository_factory(), project_id, artifact_id, revision_id=revision_id
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="精读卡不存在") from exc

    @app.get("/api/v1/projects/{project_id}/jobs/{job_id}")
    def api_get_job(project_id: str, job_id: str):
        job = job_queue_factory().repository.get(job_id)
        if job is None or job.project_id != project_id:
            raise HTTPException(status_code=404, detail="任务不存在")
        return _public_job(job)

    @app.post(
        "/api/v1/projects/{project_id}/deep-reads/{artifact_id}/fields/{field_name}/regenerations",
        status_code=202,
    )
    def api_regenerate_field(project_id: str, artifact_id: str, field_name: str):
        try:
            job = _enqueue_field(
                repository_factory(),
                job_queue_factory(),
                project_id,
                artifact_id,
                field_name,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="精读卡不存在") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=_exception_message(exc)) from exc
        return {"job": _public_job(job)}

    @app.patch(
        "/api/v1/projects/{project_id}/deep-reads/{artifact_id}/fields/{field_name}"
    )
    async def api_edit_field(
        request: Request,
        project_id: str,
        artifact_id: str,
        field_name: str,
    ):
        payload = await request.json()
        try:
            expected = _strict_int(payload.get("expected_artifact_version"))
            text = str(payload.get("text", ""))
            saved = DeepReadArtifactService(repository_factory()).edit_field(
                project_id,
                artifact_id,
                field_name,
                text,
                expected_artifact_version=expected,
            )
        except RecordVersionConflictError as exc:
            raise HTTPException(status_code=409, detail="精读卡版本冲突") from exc
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="精读卡不存在") from exc
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=_exception_message(exc)) from exc
        return _detail_context(repository_factory(), project_id, saved.artifact.id)

    @app.get("/api/v1/projects/{project_id}/deep-reads/{artifact_id}/revisions")
    def api_revisions(project_id: str, artifact_id: str):
        artifact = _scoped_artifact(repository_factory(), project_id, artifact_id)
        page = repository_factory().list_artifact_revisions(artifact.id)
        return {"items": [_public_revision(item) for item in page.items]}

    @app.get(
        "/api/v1/projects/{project_id}/deep-reads/{artifact_id}/revisions/{revision_id}/fields/{field_name}/evidence"
    )
    def api_field_evidence(
        project_id: str,
        artifact_id: str,
        revision_id: str,
        field_name: str,
    ):
        return {
            "items": _field_evidence(
                repository_factory(),
                store_factory(),
                project_id,
                artifact_id,
                revision_id,
                field_name,
            )
        }

    @app.get("/ui/projects/{project_id}/deep-reads", response_class=HTMLResponse)
    def ui_deep_reads(request: Request, project_id: str):
        repository = repository_factory()
        try:
            project = _require_project(repository, project_id)
            memberships = repository.list_project_sources(project_id, limit=200).items
            artifacts = {
                item.source_id: _artifact_summary(repository, item)
                for item in repository.list_artifacts(
                    project_id, artifact_type="deep_read", limit=200
                ).items
            }
            return _render(
                templates,
                request,
                "deep_reads.html",
                {
                    "project": project,
                    "sources": memberships,
                    "artifacts": artifacts,
                    "message": None,
                },
            )
        except KeyError as exc:
            return _render_error(
                templates, request, _exception_message(exc), status_code=404
            )

    @app.post(
        "/ui/projects/{project_id}/sources/{source_id}/deep-reads",
        response_class=HTMLResponse,
    )
    async def ui_generate(request: Request, project_id: str, source_id: str):
        try:
            await _validated_form(request)
            artifact, job = _enqueue_full(
                repository_factory(), job_queue_factory(), project_id, source_id
            )
            return _render(
                templates,
                request,
                "fragments/deep_read_action.html",
                {
                    "message": "精读任务已进入后台队列",
                    "artifact": artifact,
                    "job": _public_job(job),
                    "project_id": project_id,
                },
            )
        except HTTPException:
            raise
        except Exception as exc:
            return _render_error(
                templates, request, _exception_message(exc), status_code=400
            )

    @app.get(
        "/ui/projects/{project_id}/deep-reads/{artifact_id}",
        response_class=HTMLResponse,
    )
    def ui_detail(
        request: Request,
        project_id: str,
        artifact_id: str,
        revision_id: Optional[str] = None,
        job_id: Optional[str] = None,
    ):
        try:
            detail = _detail_context(
                repository_factory(), project_id, artifact_id, revision_id=revision_id
            )
            job = None
            if job_id:
                candidate = job_queue_factory().repository.get(job_id)
                if candidate and candidate.project_id == project_id and candidate.artifact_id == artifact_id:
                    job = _public_job(candidate)
            return _render(
                templates,
                request,
                "deep_read.html",
                {**detail, "job": job},
            )
        except KeyError:
            return _render_error(
                templates, request, "精读卡不存在", status_code=404
            )

    @app.get(
        "/ui/projects/{project_id}/deep-reads/{artifact_id}/state",
        response_class=HTMLResponse,
    )
    def ui_state(request: Request, project_id: str, artifact_id: str, job_id: str):
        try:
            detail = _detail_context(repository_factory(), project_id, artifact_id)
            job = job_queue_factory().repository.get(job_id)
            if job is None or job.project_id != project_id or job.artifact_id != artifact_id:
                raise KeyError
            return _render(
                templates,
                request,
                "fragments/deep_read_state.html",
                {**detail, "job": _public_job(job)},
            )
        except KeyError:
            return _render_error(
                templates, request, "精读任务不存在", status_code=404
            )

    @app.post(
        "/ui/projects/{project_id}/deep-reads/{artifact_id}/fields/{field_name}/edit",
        response_class=HTMLResponse,
    )
    async def ui_edit(
        request: Request, project_id: str, artifact_id: str, field_name: str
    ):
        try:
            form = await _validated_form(request)
            saved = DeepReadArtifactService(repository_factory()).edit_field(
                project_id,
                artifact_id,
                field_name,
                form.get("text", ""),
                expected_artifact_version=_form_int(
                    form, "expected_artifact_version", minimum=1
                ),
            )
            detail = _detail_context(repository_factory(), project_id, saved.artifact.id)
            return _render(
                templates,
                request,
                "fragments/deep_read_field.html",
                {
                    "field_name": field_name,
                    "field": detail["fields_by_name"][field_name],
                    "artifact": detail["artifact"],
                    "project": detail["project"],
                    "revision": detail["revision"],
                    "freshness": detail["freshness"],
                    "current": True,
                },
            )
        except RecordVersionConflictError:
            return _render_error(
                templates,
                request,
                "精读卡版本冲突，请刷新后重试",
                status_code=409,
            )
        except HTTPException:
            raise
        except Exception as exc:
            return _render_error(
                templates, request, _exception_message(exc), status_code=400
            )

    @app.post(
        "/ui/projects/{project_id}/deep-reads/{artifact_id}/fields/{field_name}/regenerate",
        response_class=HTMLResponse,
    )
    async def ui_regenerate(
        request: Request, project_id: str, artifact_id: str, field_name: str
    ):
        try:
            await _validated_form(request)
            job = _enqueue_field(
                repository_factory(),
                job_queue_factory(),
                project_id,
                artifact_id,
                field_name,
            )
            artifact = _scoped_artifact(repository_factory(), project_id, artifact_id)
            return _render(
                templates,
                request,
                "fragments/deep_read_action.html",
                {
                    "message": "本栏重新生成任务已排队",
                    "artifact": artifact,
                    "job": _public_job(job),
                    "project_id": project_id,
                },
            )
        except HTTPException:
            raise
        except Exception as exc:
            return _render_error(
                templates, request, _exception_message(exc), status_code=400
            )

    @app.get(
        "/ui/projects/{project_id}/deep-reads/{artifact_id}/revisions/{revision_id}/fields/{field_name}/evidence",
        response_class=HTMLResponse,
    )
    def ui_evidence(
        request: Request,
        project_id: str,
        artifact_id: str,
        revision_id: str,
        field_name: str,
    ):
        try:
            items = _field_evidence(
                repository_factory(),
                store_factory(),
                project_id,
                artifact_id,
                revision_id,
                field_name,
            )
            return _render(
                templates,
                request,
                "fragments/deep_read_evidence.html",
                {"items": items},
            )
        except KeyError:
            return _render_error(
                templates,
                request,
                "证据不存在或不属于当前精读卡",
                status_code=404,
            )


def _enqueue_full(repository, queue, project_id: str, source_id: str):
    _require_project(repository, project_id)
    service = DeepReadArtifactService(repository)
    artifact = service.ensure_artifact(project_id, source_id)
    job = queue.enqueue(
        "deep_read",
        {
            "project_id": project_id,
            "source_id": source_id,
            "expected_artifact_version": artifact.version,
        },
        project_id=project_id,
        artifact_id=artifact.id,
        timeout_seconds=600,
        max_attempts=2,
        idempotent=True,
        idempotency_key=f"deep-read:{artifact.id}:v{artifact.version}",
    )
    return artifact, job


def _enqueue_field(repository, queue, project_id, artifact_id, field_name):
    if field_name not in DEEP_READ_FIELD_ORDER:
        raise ValueError("未知精读字段")
    artifact = _scoped_artifact(repository, project_id, artifact_id)
    revision = repository.get_current_artifact_revision(artifact.id)
    if revision is None:
        raise ValueError("精读卡尚无 revision")
    if repository.artifact_freshness(artifact.id).stale:
        raise ValueError("来源已变化，请先完整重新生成精读卡")
    return queue.enqueue(
        "deep_read_field",
        {
            "project_id": project_id,
            "artifact_id": artifact.id,
            "field_name": field_name,
            "expected_artifact_version": artifact.version,
            "base_revision_id": revision.id,
        },
        project_id=project_id,
        artifact_id=artifact.id,
        timeout_seconds=180,
        max_attempts=2,
        idempotent=True,
        idempotency_key=f"deep-read-field:{artifact.id}:{field_name}:v{artifact.version}",
    )


def _detail_context(repository, project_id, artifact_id, *, revision_id=None):
    project = _require_project(repository, project_id)
    artifact = _scoped_artifact(repository, project_id, artifact_id)
    revision = (
        repository.get_artifact_revision(revision_id)
        if revision_id
        else repository.get_current_artifact_revision(artifact.id)
    )
    if revision is not None and revision.artifact_id != artifact.id:
        raise KeyError
    card = DeepReadCard.model_validate(revision.content) if revision else None
    fields = []
    fields_by_name = {}
    if card is not None:
        links = repository.list_artifact_evidence(revision.id)
        counts: dict[str, int] = {}
        for link in links:
            counts[link.field_path] = counts.get(link.field_path, 0) + 1
        for name, value in card.ordered_fields():
            item = {
                "name": name,
                "label": DEEP_READ_FIELD_LABELS[name],
                "text": value.text,
                "insufficient_evidence": value.insufficient_evidence,
                "evidence_count": counts.get(f"$.{name}", 0),
            }
            fields.append(item)
            fields_by_name[name] = item
    freshness = repository.artifact_freshness(artifact.id)
    return {
        "project": project,
        "artifact": artifact,
        "revision": revision,
        "revision_public": _public_revision(revision) if revision else None,
        "fields": fields,
        "fields_by_name": fields_by_name,
        "freshness": {"stale": freshness.stale, "reason": freshness.reason},
        "history": [
            _public_revision(item)
            for item in repository.list_artifact_revisions(artifact.id).items
        ],
        "current": revision_id is None,
    }


def _field_evidence(
    repository, store, project_id, artifact_id, revision_id, field_name
):
    if field_name not in DEEP_READ_FIELD_ORDER:
        raise KeyError
    _scoped_artifact(repository, project_id, artifact_id)
    revision = repository.get_artifact_revision(revision_id)
    if revision is None or revision.artifact_id != artifact_id:
        raise KeyError
    card = DeepReadCard.model_validate(revision.content)
    refs = {ref.evidence_id: ref.quote for ref in getattr(card, field_name).evidence_refs}
    links = [
        link
        for link in repository.list_artifact_evidence(revision.id)
        if link.field_path == f"$.{field_name}"
    ]
    result = []
    for link in links:
        evidence = store.get_evidence(link.evidence_id)
        if evidence is None or link.evidence_id not in refs:
            raise KeyError
        result.append(
            {
                "evidence_id": evidence.id,
                "source_title": evidence.title,
                "authors": list(evidence.authors),
                "year": evidence.year,
                "page": evidence.page if evidence.page > 0 else None,
                "quote": refs[evidence.id],
                "context": evidence.text,
                "stale": evidence.stale,
            }
        )
    return result


def _scoped_artifact(repository, project_id, artifact_id):
    artifact = repository.get_artifact(artifact_id)
    if (
        artifact is None
        or artifact.project_id != project_id
        or artifact.artifact_type != "deep_read"
    ):
        raise KeyError("精读卡不存在")
    return artifact


def _artifact_summary(repository, artifact):
    freshness = repository.artifact_freshness(artifact.id)
    return {
        "id": artifact.id,
        "source_id": artifact.source_id,
        "title": artifact.title,
        "status": artifact.status,
        "current_revision_number": artifact.current_revision_number,
        "version": artifact.version,
        "stale": freshness.stale,
    }


def _public_revision(revision):
    if revision is None:
        return None
    return {
        "id": revision.id,
        "revision_number": revision.revision_number,
        "created_by": revision.created_by,
        "model": revision.model,
        "finish_reason": revision.finish_reason,
        "prompt_version": revision.prompt_version,
        "schema_version": revision.schema_version,
        "created_at": revision.created_at,
    }


def _public_job(job):
    error = None
    if job.error_code:
        error = {
            "code": job.error_code,
            "message": _SAFE_JOB_ERRORS.get(job.error_code, "任务执行失败"),
        }
    return {
        "id": job.id,
        "status": job.status,
        "progress_current": job.progress_current,
        "progress_total": job.progress_total,
        "attempts": job.attempts,
        "max_attempts": job.max_attempts,
        "started_at": job.started_at,
        "finished_at": job.finished_at,
        "error": error,
        "terminal": job.status in {"succeeded", "failed", "cancelled", "interrupted"},
    }


def _strict_int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("expected_artifact_version 必须是正整数")
    return value
