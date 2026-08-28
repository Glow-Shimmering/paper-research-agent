"""Review outline/section API and HTMX workflow."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Callable, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError

from pragent.research import (
    ComparisonMatrix,
    ReviewOutline,
    ReviewOutlineArtifactService,
    ReviewSectionDraft,
)
from pragent.storage import RecordVersionConflictError

from .artifacts import _public_job, _public_revision, _strict_int
from .projects import (
    _exception_message,
    _form_int,
    _render,
    _render_error,
    _require_project,
    _validated_form,
)


def register_review_routes(
    app: FastAPI,
    *,
    repository_factory: Callable,
    job_queue_factory: Callable,
    store_factory: Callable,
    templates_directory: str,
) -> None:
    templates = Jinja2Templates(directory=templates_directory)

    @app.post("/api/v1/projects/{project_id}/review-outlines", status_code=202)
    def api_create_outline(project_id: str, payload: dict):
        try:
            return _enqueue_outline(
                repository_factory(),
                job_queue_factory(),
                project_id,
                (payload or {}).get("question_ids", ()),
                str((payload or {}).get("comparison_artifact_id") or ""),
                title=str((payload or {}).get("title") or "文献综述提纲"),
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=_exception_message(exc)) from exc
        except (TypeError, ValueError, ValidationError) as exc:
            raise HTTPException(status_code=400, detail=_exception_message(exc)) from exc

    @app.get("/api/v1/projects/{project_id}/review-outlines/{artifact_id}")
    def api_get_outline(project_id: str, artifact_id: str):
        try:
            return _review_context(repository_factory(), project_id, artifact_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="综述提纲不存在") from exc

    @app.patch(
        "/api/v1/projects/{project_id}/review-outlines/{artifact_id}/sections/{section_key}"
    )
    def api_edit_outline_section(
        project_id: str,
        artifact_id: str,
        section_key: str,
        payload: dict,
    ):
        try:
            saved = ReviewOutlineArtifactService(repository_factory()).edit_section(
                project_id,
                artifact_id,
                section_key,
                title=str((payload or {}).get("title") or ""),
                objective=str((payload or {}).get("objective") or ""),
                expected_artifact_version=_strict_int(
                    (payload or {}).get("expected_artifact_version")
                ),
            )
            return _review_context(repository_factory(), project_id, saved.artifact.id)
        except RecordVersionConflictError as exc:
            raise HTTPException(status_code=409, detail="综述提纲版本冲突") from exc
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=_exception_message(exc)) from exc
        except (TypeError, ValueError, ValidationError) as exc:
            raise HTTPException(status_code=400, detail=_exception_message(exc)) from exc

    @app.post(
        "/api/v1/projects/{project_id}/review-outlines/{artifact_id}/sections/{section_key}/drafts",
        status_code=202,
    )
    def api_generate_section(project_id: str, artifact_id: str, section_key: str):
        try:
            return _enqueue_section(
                repository_factory(),
                job_queue_factory(),
                project_id,
                artifact_id,
                section_key,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=_exception_message(exc)) from exc
        except (ValueError, ValidationError) as exc:
            raise HTTPException(status_code=400, detail=_exception_message(exc)) from exc

    @app.get(
        "/api/v1/projects/{project_id}/review-sections/{artifact_id}/revisions/{revision_id}/claims/{claim_key}/evidence"
    )
    def api_review_evidence(
        project_id: str,
        artifact_id: str,
        revision_id: str,
        claim_key: str,
    ):
        try:
            return {
                "items": _claim_evidence(
                    repository_factory(),
                    store_factory(),
                    project_id,
                    artifact_id,
                    revision_id,
                    claim_key,
                )
            }
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="章节证据不存在") from exc

    @app.get("/ui/projects/{project_id}/reviews", response_class=HTMLResponse)
    def ui_reviews(request: Request, project_id: str):
        try:
            repository = repository_factory()
            project = _require_project(repository, project_id)
            questions = repository.list_questions(project_id)
            comparisons = [
                item
                for item in repository.list_artifacts(
                    project_id, artifact_type="comparison", limit=200
                ).items
                if item.status == "ready"
                and not repository.artifact_freshness(item.id).stale
            ]
            outlines = repository.list_artifacts(
                project_id, artifact_type="review_outline", limit=200
            ).items
            return _render(
                templates,
                request,
                "reviews.html",
                {
                    "project": project,
                    "questions": questions,
                    "comparisons": comparisons,
                    "outlines": outlines,
                },
            )
        except KeyError as exc:
            return _render_error(
                templates, request, _exception_message(exc), status_code=404
            )

    @app.post("/ui/projects/{project_id}/reviews", response_class=HTMLResponse)
    async def ui_create_outline(request: Request, project_id: str):
        form = await _validated_form(request)
        try:
            result = _enqueue_outline(
                repository_factory(),
                job_queue_factory(),
                project_id,
                form.getlist("question_ids"),
                form.get("comparison_artifact_id", ""),
                title=form.get("title", "文献综述提纲"),
            )
            return _render(
                templates,
                request,
                "fragments/review_action.html",
                {"project_id": project_id, **result},
            )
        except (KeyError, TypeError, ValueError, ValidationError) as exc:
            return _render_error(
                templates, request, _exception_message(exc), status_code=400
            )

    @app.get(
        "/ui/projects/{project_id}/reviews/jobs/{job_id}",
        response_class=HTMLResponse,
    )
    def ui_review_job(request: Request, project_id: str, job_id: str):
        job = job_queue_factory().repository.get(job_id)
        if job is None or job.project_id != project_id:
            return _render_error(templates, request, "任务不存在", status_code=404)
        result = job.result or {}
        return _render(
            templates,
            request,
            "fragments/review_action.html",
            {
                "project_id": project_id,
                "status": "queued",
                "job": _public_job(job),
                "artifact_id": result.get("artifact_id"),
                "outline_artifact_id": job.payload.get("outline_artifact_id"),
            },
        )

    @app.get(
        "/ui/projects/{project_id}/reviews/{artifact_id}",
        response_class=HTMLResponse,
    )
    def ui_review(request: Request, project_id: str, artifact_id: str):
        try:
            return _render(
                templates,
                request,
                "review.html",
                _review_context(repository_factory(), project_id, artifact_id),
            )
        except KeyError:
            return _render_error(
                templates, request, "综述提纲不存在", status_code=404
            )

    @app.post(
        "/ui/projects/{project_id}/reviews/{artifact_id}/sections/{section_key}/edit",
        response_class=HTMLResponse,
    )
    async def ui_edit_outline_section(
        request: Request,
        project_id: str,
        artifact_id: str,
        section_key: str,
    ):
        form = await _validated_form(request)
        try:
            saved = ReviewOutlineArtifactService(repository_factory()).edit_section(
                project_id,
                artifact_id,
                section_key,
                title=form.get("title", ""),
                objective=form.get("objective", ""),
                expected_artifact_version=_form_int(
                    form, "expected_artifact_version", minimum=1
                ),
            )
            return _render(
                templates,
                request,
                "fragments/review_state.html",
                _review_context(repository_factory(), project_id, saved.artifact.id),
            )
        except RecordVersionConflictError:
            return _render_error(
                templates, request, "综述提纲版本冲突，请刷新后重试", status_code=409
            )
        except (KeyError, TypeError, ValueError, ValidationError) as exc:
            return _render_error(
                templates, request, _exception_message(exc), status_code=400
            )

    @app.post(
        "/ui/projects/{project_id}/reviews/{artifact_id}/sections/{section_key}/drafts",
        response_class=HTMLResponse,
    )
    async def ui_generate_section(
        request: Request,
        project_id: str,
        artifact_id: str,
        section_key: str,
    ):
        await _validated_form(request)
        try:
            result = _enqueue_section(
                repository_factory(),
                job_queue_factory(),
                project_id,
                artifact_id,
                section_key,
            )
            return _render(
                templates,
                request,
                "fragments/review_action.html",
                {
                    "project_id": project_id,
                    "outline_artifact_id": artifact_id,
                    **result,
                },
            )
        except (KeyError, ValueError, ValidationError) as exc:
            return _render_error(
                templates, request, _exception_message(exc), status_code=400
            )

    @app.get(
        "/ui/projects/{project_id}/review-sections/{artifact_id}/revisions/{revision_id}/claims/{claim_key}/evidence",
        response_class=HTMLResponse,
    )
    def ui_review_evidence(
        request: Request,
        project_id: str,
        artifact_id: str,
        revision_id: str,
        claim_key: str,
    ):
        try:
            return _render(
                templates,
                request,
                "fragments/deep_read_evidence.html",
                {
                    "items": _claim_evidence(
                        repository_factory(),
                        store_factory(),
                        project_id,
                        artifact_id,
                        revision_id,
                        claim_key,
                    )
                },
            )
        except KeyError:
            return _render_error(
                templates, request, "章节证据不存在", status_code=404
            )


def _enqueue_outline(repository, queue, project_id, question_ids, comparison_id, *, title):
    _require_project(repository, project_id)
    if not isinstance(question_ids, (list, tuple)):
        raise ValueError("question_ids 必须是数组")
    selected_questions = tuple(str(item).strip() for item in question_ids)
    known_questions = {item.id: item for item in repository.list_questions(project_id)}
    if not 1 <= len(selected_questions) <= 20 or len(selected_questions) != len(
        set(selected_questions)
    ):
        raise ValueError("必须选择 1–20 个不重复研究问题")
    if set(selected_questions) - set(known_questions):
        raise ValueError("研究问题必须属于当前项目")
    comparison = _scoped_artifact(repository, project_id, comparison_id, "comparison")
    if comparison.status != "ready" or repository.artifact_freshness(comparison.id).stale:
        raise ValueError("比较矩阵未完成或已过期")
    revision = repository.get_current_artifact_revision(comparison.id)
    if revision is None:
        raise ValueError("比较矩阵尚无 revision")
    matrix = ComparisonMatrix.model_validate(revision.content)
    normalized_title = str(title).strip() or "文献综述提纲"
    canonical = json.dumps(
        {
            "project_id": project_id,
            "question_ids": selected_questions,
            "question_versions": {
                item: known_questions[item].version for item in selected_questions
            },
            "source_ids": matrix.source_ids,
            "comparison_artifact_id": comparison.id,
            "comparison_revision_id": revision.id,
            "project_fingerprint": repository.project_source_fingerprint(project_id),
            "title": normalized_title,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    job = queue.enqueue(
        "review_outline",
        json.loads(canonical),
        project_id=project_id,
        timeout_seconds=600,
        max_attempts=2,
        idempotent=True,
        idempotency_key="review-outline:" + hashlib.sha256(canonical.encode()).hexdigest(),
    )
    return {"status": "queued", "job": _public_job(job)}


def _enqueue_section(repository, queue, project_id, outline_id, section_key):
    context = _review_context(repository, project_id, outline_id)
    if context["derived_stale"]:
        raise ValueError("综述提纲输入已变化，请先重新生成")
    if section_key not in {item.key for item in context["outline"].sections}:
        raise KeyError("综述提纲 section 不存在")
    existing_count = len(
        [
            item
            for item in repository.list_artifacts(
                project_id, artifact_type="review_section", limit=200
            ).items
            if _section_matches(repository, item, outline_id, section_key)
        ]
    )
    revision = context["revision"]
    payload = {
        "project_id": project_id,
        "outline_artifact_id": outline_id,
        "outline_revision_id": revision.id,
        "section_key": section_key,
        "generation_number": existing_count + 1,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    job = queue.enqueue(
        "review_section",
        payload,
        project_id=project_id,
        timeout_seconds=600,
        max_attempts=2,
        idempotent=True,
        idempotency_key="review-section:" + hashlib.sha256(canonical.encode()).hexdigest(),
    )
    return {"status": "queued", "job": _public_job(job)}


def _review_context(repository, project_id, artifact_id):
    project = _require_project(repository, project_id)
    artifact = _scoped_artifact(repository, project_id, artifact_id, "review_outline")
    revision = repository.get_current_artifact_revision(artifact.id)
    if revision is None:
        raise KeyError("综述提纲尚无 revision")
    outline = ReviewOutline.model_validate(revision.content)
    questions = {item.id: item for item in repository.list_questions(project_id)}
    questions_stale = any(
        item.id not in questions
        or questions[item.id].version != item.version
        or questions[item.id].question != item.question
        for item in outline.research_questions
    )
    comparison = repository.get_artifact(outline.comparison_artifact_id)
    comparison_revision = repository.get_current_artifact_revision(
        outline.comparison_artifact_id
    )
    comparison_stale = (
        comparison is None
        or comparison.project_id != project_id
        or comparison.artifact_type != "comparison"
        or comparison.status != "ready"
        or comparison_revision is None
        or comparison_revision.id != outline.comparison_revision_id
        or repository.artifact_freshness(comparison.id).stale
    )
    drafts = {}
    for candidate in repository.list_artifacts(
        project_id, artifact_type="review_section", limit=200
    ).items:
        current = repository.get_current_artifact_revision(candidate.id)
        if current is None:
            continue
        try:
            draft = ReviewSectionDraft.model_validate(current.content)
        except ValidationError:
            continue
        if (
            draft.outline_artifact_id == artifact.id
            and draft.outline_revision_id == revision.id
            and draft.section_key not in drafts
        ):
            drafts[draft.section_key] = {
                "artifact": candidate,
                "revision": current,
                "draft": draft,
            }
    return {
        "project": project,
        "artifact": artifact,
        "revision": revision,
        "revision_public": _public_revision(revision),
        "outline": outline,
        "drafts": drafts,
        "history": [
            _public_revision(item)
            for item in repository.list_artifact_revisions(artifact.id).items
        ],
        "freshness": repository.artifact_freshness(artifact.id),
        "derived_stale": questions_stale or comparison_stale,
    }


def _claim_evidence(repository, store, project_id, artifact_id, revision_id, claim_key):
    _scoped_artifact(repository, project_id, artifact_id, "review_section")
    revision = repository.get_artifact_revision(revision_id)
    if revision is None or revision.artifact_id != artifact_id:
        raise KeyError
    draft = ReviewSectionDraft.model_validate(revision.content)
    claim_index = next(
        (index for index, item in enumerate(draft.claims) if item.key == claim_key),
        None,
    )
    if claim_index is None:
        raise KeyError
    claim = draft.claims[claim_index]
    tokens = {item.evidence_id: item for item in claim.citation_tokens}
    result = []
    for link in repository.list_artifact_evidence(revision.id):
        if link.field_path != f"$.claims.{claim_index}":
            continue
        evidence = store.get_evidence(link.evidence_id)
        token = tokens.get(link.evidence_id)
        if evidence is None or token is None:
            raise KeyError
        result.append(
            {
                "evidence_id": evidence.id,
                "source_title": evidence.title,
                "authors": list(evidence.authors),
                "year": evidence.year,
                "page": evidence.page if evidence.page > 0 else None,
                "quote": token.quote,
                "context": evidence.text,
                "stale": evidence.stale,
            }
        )
    if len(result) != len(tokens):
        raise KeyError
    return result


def _section_matches(repository, artifact, outline_id, section_key):
    revision = repository.get_current_artifact_revision(artifact.id)
    if revision is None:
        return False
    try:
        draft = ReviewSectionDraft.model_validate(revision.content)
    except ValidationError:
        return False
    return draft.outline_artifact_id == outline_id and draft.section_key == section_key


def _scoped_artifact(repository, project_id, artifact_id, artifact_type):
    artifact = repository.get_artifact(artifact_id)
    if (
        artifact is None
        or artifact.project_id != project_id
        or artifact.artifact_type != artifact_type
        or artifact.source_id is not None
    ):
        raise KeyError("artifact 不存在")
    return artifact
