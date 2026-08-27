"""Project-scoped comparison matrix JSON API and HTMX pages."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Callable, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError

from pragent.research import (
    DEEP_READ_FIELD_ORDER,
    ComparisonArtifactService,
    ComparisonDimension,
    ComparisonMatrix,
    ComparisonPrerequisiteError,
    ComparisonWorkflow,
)
from pragent.storage import RecordVersionConflictError

from .artifacts import _enqueue_full, _public_job, _public_revision, _strict_int
from .projects import (
    _exception_message,
    _form_int,
    _render,
    _render_error,
    _require_project,
    _validated_form,
)


def register_comparison_routes(
    app: FastAPI,
    *,
    repository_factory: Callable,
    job_queue_factory: Callable,
    store_factory: Callable,
    templates_directory: str,
) -> None:
    templates = Jinja2Templates(directory=templates_directory)

    @app.get("/api/v1/projects/{project_id}/comparisons")
    def api_list_comparisons(project_id: str):
        repository = repository_factory()
        _require_project(repository, project_id)
        page = repository.list_artifacts(
            project_id, artifact_type="comparison", limit=200
        )
        return {"items": [_comparison_summary(repository, item) for item in page.items]}

    @app.post("/api/v1/projects/{project_id}/comparisons", status_code=202)
    def api_create_comparison(project_id: str, payload: dict):
        try:
            return _start_comparison(
                repository_factory(),
                job_queue_factory(),
                project_id,
                (payload or {}).get("source_ids", ()),
                title=str((payload or {}).get("title") or "跨论文比较矩阵"),
                custom_dimensions=(payload or {}).get("custom_dimensions", ()),
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=_exception_message(exc)) from exc
        except (TypeError, ValueError, ValidationError) as exc:
            raise HTTPException(status_code=400, detail=_exception_message(exc)) from exc

    @app.get("/api/v1/projects/{project_id}/comparisons/{artifact_id}")
    def api_get_comparison(
        project_id: str,
        artifact_id: str,
        revision_id: Optional[str] = None,
    ):
        try:
            return _comparison_context(
                repository_factory(), project_id, artifact_id, revision_id=revision_id
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="比较矩阵不存在") from exc

    @app.patch(
        "/api/v1/projects/{project_id}/comparisons/{artifact_id}/cells/{source_id}/{dimension_key}"
    )
    def api_edit_cell(
        project_id: str,
        artifact_id: str,
        source_id: str,
        dimension_key: str,
        payload: dict,
    ):
        try:
            insufficient = (payload or {}).get("insufficient_evidence")
            if insufficient is not None and not isinstance(insufficient, bool):
                raise ValueError("insufficient_evidence 必须是布尔值")
            saved = ComparisonArtifactService(repository_factory()).edit_cell(
                project_id,
                artifact_id,
                source_id,
                dimension_key,
                str((payload or {}).get("summary") or ""),
                expected_artifact_version=_strict_int(
                    (payload or {}).get("expected_artifact_version")
                ),
                insufficient_evidence=insufficient,
            )
            return _comparison_context(
                repository_factory(), project_id, saved.artifact.id
            )
        except RecordVersionConflictError as exc:
            raise HTTPException(status_code=409, detail="比较矩阵版本冲突") from exc
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=_exception_message(exc)) from exc
        except (TypeError, ValueError, ValidationError) as exc:
            raise HTTPException(status_code=400, detail=_exception_message(exc)) from exc

    @app.get(
        "/api/v1/projects/{project_id}/comparisons/{artifact_id}/revisions"
    )
    def api_comparison_revisions(project_id: str, artifact_id: str):
        artifact = _scoped_comparison(repository_factory(), project_id, artifact_id)
        return {
            "items": [
                _public_revision(item)
                for item in repository_factory().list_artifact_revisions(
                    artifact.id
                ).items
            ]
        }

    @app.get(
        "/api/v1/projects/{project_id}/comparisons/{artifact_id}/revisions/{revision_id}/cells/{source_id}/{dimension_key}/evidence"
    )
    def api_comparison_evidence(
        project_id: str,
        artifact_id: str,
        revision_id: str,
        source_id: str,
        dimension_key: str,
    ):
        try:
            items = _cell_evidence(
                repository_factory(),
                store_factory(),
                project_id,
                artifact_id,
                revision_id,
                source_id,
                dimension_key,
            )
            return {"items": items}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="比较证据不存在") from exc

    @app.get("/ui/projects/{project_id}/comparisons", response_class=HTMLResponse)
    def ui_comparisons(request: Request, project_id: str):
        try:
            repository = repository_factory()
            project = _require_project(repository, project_id)
            memberships = repository.list_project_sources(project_id, limit=200).items
            artifacts = repository.list_artifacts(
                project_id, artifact_type="comparison", limit=200
            ).items
            return _render(
                templates,
                request,
                "comparisons.html",
                {
                    "project": project,
                    "sources": memberships,
                    "comparisons": [
                        _comparison_summary(repository, item) for item in artifacts
                    ],
                },
            )
        except KeyError as exc:
            return _render_error(
                templates, request, _exception_message(exc), status_code=404
            )

    @app.post("/ui/projects/{project_id}/comparisons", response_class=HTMLResponse)
    async def ui_create_comparison(request: Request, project_id: str):
        form = await _validated_form(request)
        try:
            result = _start_comparison(
                repository_factory(),
                job_queue_factory(),
                project_id,
                form.getlist("source_ids"),
                title=form.get("title", "跨论文比较矩阵"),
                custom_dimensions=_parse_custom_dimension_lines(
                    form.get("custom_dimensions", "")
                ),
            )
            return _render(
                templates,
                request,
                "fragments/comparison_action.html",
                {"project_id": project_id, **result},
            )
        except (KeyError, TypeError, ValueError, ValidationError) as exc:
            return _render_error(
                templates, request, _exception_message(exc), status_code=400
            )

    @app.get(
        "/ui/projects/{project_id}/comparisons/jobs/{job_id}",
        response_class=HTMLResponse,
    )
    def ui_comparison_job(request: Request, project_id: str, job_id: str):
        job = job_queue_factory().repository.get(job_id)
        if job is None or job.project_id != project_id:
            return _render_error(templates, request, "任务不存在", status_code=404)
        return _render(
            templates,
            request,
            "fragments/comparison_action.html",
            {
                "project_id": project_id,
                "status": "queued",
                "job": _public_job(job),
                "artifact_id": (job.result or {}).get("artifact_id"),
            },
        )

    @app.get(
        "/ui/projects/{project_id}/comparisons/{artifact_id}",
        response_class=HTMLResponse,
    )
    def ui_comparison(
        request: Request,
        project_id: str,
        artifact_id: str,
        revision_id: Optional[str] = None,
    ):
        try:
            return _render(
                templates,
                request,
                "comparison.html",
                _comparison_context(
                    repository_factory(),
                    project_id,
                    artifact_id,
                    revision_id=revision_id,
                ),
            )
        except KeyError:
            return _render_error(
                templates, request, "比较矩阵不存在", status_code=404
            )

    @app.post(
        "/ui/projects/{project_id}/comparisons/{artifact_id}/cells/{source_id}/{dimension_key}/edit",
        response_class=HTMLResponse,
    )
    async def ui_edit_cell(
        request: Request,
        project_id: str,
        artifact_id: str,
        source_id: str,
        dimension_key: str,
    ):
        form = await _validated_form(request)
        try:
            insufficient = form.get("insufficient_evidence") == "on"
            saved = ComparisonArtifactService(repository_factory()).edit_cell(
                project_id,
                artifact_id,
                source_id,
                dimension_key,
                form.get("summary", ""),
                expected_artifact_version=_form_int(
                    form, "expected_artifact_version", minimum=1
                ),
                insufficient_evidence=insufficient,
            )
            context = _comparison_context(
                repository_factory(), project_id, saved.artifact.id
            )
            return _render(
                templates,
                request,
                "fragments/comparison_state.html",
                context,
            )
        except RecordVersionConflictError:
            return _render_error(
                templates, request, "比较矩阵版本冲突，请刷新后重试", status_code=409
            )
        except (KeyError, TypeError, ValueError, ValidationError) as exc:
            return _render_error(
                templates, request, _exception_message(exc), status_code=400
            )

    @app.get(
        "/ui/projects/{project_id}/comparisons/{artifact_id}/revisions/{revision_id}/cells/{source_id}/{dimension_key}/evidence",
        response_class=HTMLResponse,
    )
    def ui_comparison_evidence(
        request: Request,
        project_id: str,
        artifact_id: str,
        revision_id: str,
        source_id: str,
        dimension_key: str,
    ):
        try:
            items = _cell_evidence(
                repository_factory(),
                store_factory(),
                project_id,
                artifact_id,
                revision_id,
                source_id,
                dimension_key,
            )
            return _render(
                templates,
                request,
                "fragments/deep_read_evidence.html",
                {"items": items},
            )
        except KeyError:
            return _render_error(
                templates, request, "比较证据不存在", status_code=404
            )


def _start_comparison(
    repository,
    queue,
    project_id: str,
    source_ids,
    *,
    title: str,
    custom_dimensions,
) -> dict[str, Any]:
    if not isinstance(source_ids, (list, tuple)):
        raise ValueError("source_ids 必须是数组")
    selected = tuple(str(item).strip() for item in source_ids)
    dimensions = tuple(
        item
        if isinstance(item, ComparisonDimension)
        else ComparisonDimension.model_validate(item)
        for item in custom_dimensions
    )
    if len(dimensions) > 20:
        raise ValueError("自定义比较维度不能超过 20 个")
    keys = [item.key for item in dimensions]
    if len(keys) != len(set(keys)) or set(keys) & set(DEEP_READ_FIELD_ORDER):
        raise ValueError("自定义比较维度 key 不能重复或覆盖默认维度")
    if any(item.source_field is not None for item in dimensions):
        raise ValueError("自定义比较维度不能设置 source_field")
    normalized_title = str(title).strip() or "跨论文比较矩阵"
    if len(normalized_title) > 200:
        raise ValueError("比较标题不能超过 200 个字符")
    try:
        ComparisonWorkflow(repository).validate_prerequisites(project_id, selected)
    except ComparisonPrerequisiteError as exc:
        jobs = []
        for source_id in (*exc.missing_source_ids, *exc.stale_source_ids):
            _artifact, job = _enqueue_full(repository, queue, project_id, source_id)
            jobs.append(_public_job(job))
        return {
            "status": "prerequisites_queued",
            "missing_source_ids": list(exc.missing_source_ids),
            "stale_source_ids": list(exc.stale_source_ids),
            "jobs": jobs,
        }
    canonical = json.dumps(
        {
            "project_id": project_id,
            "source_ids": selected,
            "title": normalized_title,
            "custom_dimensions": [item.model_dump(mode="json") for item in dimensions],
            "project_fingerprint": repository.project_source_fingerprint(project_id),
            "deep_read_revision_ids": {
                source_id: repository.get_current_artifact_revision(
                    repository.get_source_artifact(
                        project_id, source_id, "deep_read"
                    ).id
                ).id
                for source_id in selected
            },
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    job = queue.enqueue(
        "comparison",
        json.loads(canonical),
        project_id=project_id,
        timeout_seconds=900,
        max_attempts=2,
        idempotent=True,
        idempotency_key="comparison:" + hashlib.sha256(canonical.encode()).hexdigest(),
    )
    return {"status": "queued", "job": _public_job(job)}


def _comparison_context(repository, project_id, artifact_id, *, revision_id=None):
    project = _require_project(repository, project_id)
    artifact = _scoped_comparison(repository, project_id, artifact_id)
    revision = (
        repository.get_artifact_revision(revision_id)
        if revision_id
        else repository.get_current_artifact_revision(artifact.id)
    )
    if revision is None or revision.artifact_id != artifact.id:
        raise KeyError("比较 revision 不存在")
    matrix = ComparisonMatrix.model_validate(revision.content)
    memberships = {
        item.source.id: item.source
        for item in repository.list_project_sources(project_id, limit=200).items
    }
    by_pair = {(cell.source_id, cell.dimension_key): cell for cell in matrix.cells}
    rows = []
    for source_id in matrix.source_ids:
        source = memberships.get(source_id)
        cells = []
        for dimension in matrix.dimensions:
            cell = by_pair[(source_id, dimension.key)]
            cells.append(
                {
                    **cell.model_dump(mode="json"),
                    "dimension_label": dimension.label,
                    "evidence_count": len(cell.evidence_refs),
                }
            )
        rows.append(
            {
                "source_id": source_id,
                "source_title": source.title if source else source_id,
                "cells": cells,
            }
        )
    freshness = repository.artifact_freshness(artifact.id)
    return {
        "project": project,
        "artifact": artifact,
        "revision": revision,
        "revision_public": _public_revision(revision),
        "matrix": matrix,
        "rows": rows,
        "freshness": {"stale": freshness.stale, "reason": freshness.reason},
        "history": [
            _public_revision(item)
            for item in repository.list_artifact_revisions(artifact.id).items
        ],
        "current": revision_id is None,
    }


def _cell_evidence(
    repository,
    store,
    project_id,
    artifact_id,
    revision_id,
    source_id,
    dimension_key,
):
    _scoped_comparison(repository, project_id, artifact_id)
    revision = repository.get_artifact_revision(revision_id)
    if revision is None or revision.artifact_id != artifact_id:
        raise KeyError
    matrix = ComparisonMatrix.model_validate(revision.content)
    cell = next(
        (
            item
            for item in matrix.cells
            if item.source_id == source_id and item.dimension_key == dimension_key
        ),
        None,
    )
    if cell is None:
        raise KeyError
    quotes = {item.evidence_id: item.quote for item in cell.evidence_refs}
    field_path = f"$.cells.{source_id}.{dimension_key}"
    result = []
    for link in repository.list_artifact_evidence(revision.id):
        if link.field_path != field_path:
            continue
        evidence = store.get_evidence(link.evidence_id)
        if evidence is None or link.evidence_id not in quotes:
            raise KeyError
        result.append(
            {
                "evidence_id": evidence.id,
                "source_title": evidence.title,
                "authors": list(evidence.authors),
                "year": evidence.year,
                "page": evidence.page if evidence.page > 0 else None,
                "quote": quotes[evidence.id],
                "context": evidence.text,
                "stale": evidence.stale,
            }
        )
    if len(result) != len(quotes):
        raise KeyError
    return result


def _scoped_comparison(repository, project_id, artifact_id):
    artifact = repository.get_artifact(artifact_id)
    if (
        artifact is None
        or artifact.project_id != project_id
        or artifact.artifact_type != "comparison"
        or artifact.source_id is not None
    ):
        raise KeyError("比较矩阵不存在")
    return artifact


def _comparison_summary(repository, artifact):
    freshness = repository.artifact_freshness(artifact.id)
    return {
        "id": artifact.id,
        "title": artifact.title,
        "status": artifact.status,
        "current_revision_number": artifact.current_revision_number,
        "version": artifact.version,
        "stale": freshness.stale,
    }


def _parse_custom_dimension_lines(raw: str) -> tuple[ComparisonDimension, ...]:
    result = []
    for line_number, raw_line in enumerate(str(raw).splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = [item.strip() for item in line.split("|", 2)]
        if len(parts) < 2:
            raise ValueError(f"自定义维度第 {line_number} 行应为 key|标签|说明")
        result.append(
            ComparisonDimension(
                key=parts[0],
                label=parts[1],
                description=parts[2] if len(parts) > 2 else "",
            )
        )
    return tuple(result)
