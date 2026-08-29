"""Phase 2 project workspace JSON API and HTMX vertical slice."""

from __future__ import annotations

import secrets
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import parse_qs

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates

from pragent.models import (
    ProjectSourceMembership,
    ResearchProject,
    ResearchQuestion,
)
from pragent.storage import RecordVersionConflictError, SourceIdentityConflictError

_CSRF_COOKIE = "pra_csrf"
_MAX_FORM_BYTES = 1_000_000


def register_project_routes(
    app: FastAPI,
    *,
    store_factory: Callable,
    repository_factory: Callable,
    templates_directory: str,
) -> None:
    templates = Jinja2Templates(directory=templates_directory)

    # ---------- JSON API ----------

    @app.get("/api/v1/projects")
    def api_projects(
        q: str = Query("", max_length=500),
        status: Optional[str] = Query(None),
        limit: int = Query(50, ge=1, le=200),
        offset: int = Query(0, ge=0),
    ):
        try:
            page = repository_factory().list_projects(
                q=q or None, status=status, limit=limit, offset=offset
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "total": page.total,
            "limit": page.limit,
            "offset": page.offset,
            "items": [_project_dict(item) for item in page.items],
        }

    @app.post("/api/v1/projects", status_code=201)
    def api_create_project(payload: dict):
        try:
            project = repository_factory().create_project(
                str((payload or {}).get("title") or ""),
                description=str((payload or {}).get("description") or ""),
                default_language=str(
                    (payload or {}).get("default_language") or "zh-CN"
                ),
                citation_style=str(
                    (payload or {}).get("citation_style")
                    or "gb-t-7714-2015-numeric"
                ),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _project_dict(project)

    @app.get("/api/v1/projects/{project_id}")
    def api_project(project_id: str):
        return _project_dict(_require_project(repository_factory(), project_id))

    @app.patch("/api/v1/projects/{project_id}")
    def api_update_project(project_id: str, payload: dict):
        expected_version = _required_int(payload, "expected_version")
        fields = {
            key: payload[key]
            for key in (
                "title",
                "description",
                "default_language",
                "citation_style",
                "status",
            )
            if key in (payload or {})
        }
        try:
            project = repository_factory().update_project(
                project_id, expected_version=expected_version, **fields
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=_exception_message(exc)) from exc
        except RecordVersionConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _project_dict(project)

    @app.get("/api/v1/projects/{project_id}/questions")
    def api_questions(project_id: str):
        _require_project(repository_factory(), project_id)
        return {
            "items": [
                _question_dict(question)
                for question in repository_factory().list_questions(project_id)
            ]
        }

    @app.post("/api/v1/projects/{project_id}/questions", status_code=201)
    def api_create_question(project_id: str, payload: dict):
        try:
            question = repository_factory().create_question(
                project_id,
                str((payload or {}).get("question") or ""),
                position=(payload or {}).get("position"),
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=_exception_message(exc)) from exc
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _question_dict(question)

    @app.patch("/api/v1/projects/{project_id}/questions/{question_id}")
    def api_update_question(project_id: str, question_id: str, payload: dict):
        repository = repository_factory()
        _require_project(repository, project_id)
        known_ids = {item.id for item in repository.list_questions(project_id)}
        if question_id not in known_ids:
            raise HTTPException(status_code=404, detail="研究问题不属于当前项目")
        expected_version = _required_int(payload, "expected_version")
        fields = {
            key: payload[key]
            for key in ("question", "position")
            if key in (payload or {})
        }
        try:
            question = repository.update_question(
                question_id,
                expected_version=expected_version,
                project_id=project_id,
                **fields,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=_exception_message(exc)) from exc
        except RecordVersionConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _question_dict(question)

    @app.delete("/api/v1/projects/{project_id}/questions/{question_id}", status_code=204)
    def api_delete_question(
        project_id: str,
        question_id: str,
        expected_version: int = Query(..., ge=1),
    ):
        _require_project(repository_factory(), project_id)
        questions = {
            question.id: question
            for question in repository_factory().list_questions(project_id)
        }
        if question_id not in questions:
            raise HTTPException(status_code=404, detail="研究问题不属于当前项目")
        try:
            repository_factory().delete_question(
                question_id,
                expected_version=expected_version,
                project_id=project_id,
            )
        except RecordVersionConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return Response(status_code=204)

    @app.get("/api/v1/projects/{project_id}/sources")
    def api_project_sources(
        project_id: str,
        q: str = Query("", max_length=500),
        limit: int = Query(50, ge=1, le=200),
        offset: int = Query(0, ge=0),
    ):
        _require_project(repository_factory(), project_id)
        page = repository_factory().list_project_sources(
            project_id, q=q or None, limit=limit, offset=offset
        )
        return {
            "total": page.total,
            "limit": page.limit,
            "offset": page.offset,
            "items": [_membership_dict(item) for item in page.items],
        }

    @app.get("/api/v1/projects/{project_id}/available-papers")
    def api_available_papers(
        project_id: str,
        q: str = Query("", max_length=500),
        limit: int = Query(50, ge=1, le=200),
        offset: int = Query(0, ge=0),
    ):
        _require_project(repository_factory(), project_id)
        total, papers = store_factory().list_papers_with_chunk_counts(
            q or None, limit, offset
        )
        selected = {
            membership.source.indexed_paper_id
            for membership in repository_factory()
            .list_project_sources(project_id, limit=200, offset=0)
            .items
            if membership.source.indexed_paper_id is not None
        }
        return {
            "total": total,
            "limit": limit,
            "offset": offset,
            "items": [
                _paper_dict(paper, chunk_count, selected=paper.id in selected)
                for paper, chunk_count in papers
            ],
        }

    @app.post("/api/v1/projects/{project_id}/sources", status_code=201)
    def api_add_project_paper(project_id: str, payload: dict):
        paper_id = _required_int(payload, "paper_id")
        try:
            membership = repository_factory().add_paper_to_project(
                project_id, paper_id
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=_exception_message(exc)) from exc
        except (SourceIdentityConflictError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return _membership_dict(membership)

    # ---------- server-rendered HTMX UI ----------

    @app.get("/ui/projects", response_class=HTMLResponse)
    def ui_projects(request: Request):
        projects = repository_factory().list_projects(limit=200, offset=0).items
        return _render(
            templates,
            request,
            "projects.html",
            {"projects": projects},
        )

    @app.post("/ui/projects")
    async def ui_create_project(request: Request):
        form = await _validated_form(request)
        try:
            project = repository_factory().create_project(
                form.get("title", ""),
                description=form.get("description", ""),
            )
        except ValueError as exc:
            return _render_error(templates, request, str(exc), status_code=400)
        location = f"/ui/projects/{project.id}"
        if _is_htmx(request):
            return Response(status_code=204, headers={"HX-Redirect": location})
        return RedirectResponse(location, status_code=303)

    @app.get("/ui/projects/{project_id}", response_class=HTMLResponse)
    def ui_project_workspace(request: Request, project_id: str):
        try:
            context = _workspace_context(
                repository_factory(), store_factory(), project_id
            )
        except KeyError as exc:
            return _render_error(
                templates,
                request,
                _exception_message(exc),
                status_code=404,
            )
        return _render(templates, request, "project_workspace.html", context)

    @app.post("/ui/projects/{project_id}/questions")
    async def ui_create_question(request: Request, project_id: str):
        form = await _validated_form(request)
        try:
            repository = repository_factory()
            repository.create_question(project_id, form.get("question", ""))
            return _question_response(templates, request, repository, project_id)
        except KeyError as exc:
            return _render_error(
                templates, request, _exception_message(exc), status_code=404
            )
        except ValueError as exc:
            return _render_error(templates, request, str(exc), status_code=400)

    @app.post("/ui/projects/{project_id}/questions/{question_id}")
    async def ui_update_question(
        request: Request, project_id: str, question_id: str
    ):
        form = await _validated_form(request)
        try:
            expected_version = _form_int(form, "expected_version", minimum=1)
            position = _form_int(form, "position", minimum=0)
            repository = repository_factory()
            known_ids = {item.id for item in repository.list_questions(project_id)}
            if question_id not in known_ids:
                raise KeyError("研究问题不属于当前项目")
            repository.update_question(
                question_id,
                expected_version=expected_version,
                question=form.get("question", ""),
                position=position,
                project_id=project_id,
            )
            return _question_response(templates, request, repository, project_id)
        except KeyError as exc:
            return _render_error(
                templates, request, _exception_message(exc), status_code=404
            )
        except RecordVersionConflictError as exc:
            return _render_error(templates, request, str(exc), status_code=409)
        except ValueError as exc:
            return _render_error(templates, request, str(exc), status_code=400)

    @app.post("/ui/projects/{project_id}/questions/{question_id}/delete")
    async def ui_delete_question(
        request: Request, project_id: str, question_id: str
    ):
        form = await _validated_form(request)
        try:
            expected_version = _form_int(form, "expected_version", minimum=1)
            repository = repository_factory()
            known_ids = {
                item.id for item in repository.list_questions(project_id)
            }
            if question_id not in known_ids:
                raise KeyError("研究问题不属于当前项目")
            repository.delete_question(
                question_id,
                expected_version=expected_version,
                project_id=project_id,
            )
            return _question_response(templates, request, repository, project_id)
        except KeyError as exc:
            return _render_error(
                templates, request, _exception_message(exc), status_code=404
            )
        except RecordVersionConflictError as exc:
            return _render_error(templates, request, str(exc), status_code=409)
        except ValueError as exc:
            return _render_error(templates, request, str(exc), status_code=400)

    @app.post("/ui/projects/{project_id}/sources")
    async def ui_add_project_paper(request: Request, project_id: str):
        form = await _validated_form(request)
        try:
            paper_id = _form_int(form, "paper_id", minimum=1)
            repository = repository_factory()
            repository.add_paper_to_project(project_id, paper_id)
            return _sources_response(
                templates, request, repository, store_factory(), project_id
            )
        except KeyError as exc:
            return _render_error(
                templates, request, _exception_message(exc), status_code=404
            )
        except (SourceIdentityConflictError, ValueError) as exc:
            return _render_error(templates, request, str(exc), status_code=409)

    # ---------- research notes ----------

    @app.get("/api/v1/projects/{project_id}/notes")
    def api_project_notes(
        project_id: str,
        source_id: Optional[str] = Query(None, max_length=128),
        evidence_id: Optional[str] = Query(None, max_length=128),
        limit: int = Query(50, ge=1, le=200),
        offset: int = Query(0, ge=0),
    ):
        repository = repository_factory()
        _require_project(repository, project_id)
        page = repository.list_notes(
            project_id, source_id=source_id, evidence_id=evidence_id,
            limit=limit, offset=offset,
        )
        return {
            "total": page.total,
            "limit": page.limit,
            "offset": page.offset,
            "items": [_note_dict(note) for note in page.items],
        }

    @app.post("/api/v1/projects/{project_id}/notes", status_code=201)
    def api_create_note(project_id: str, payload: dict):
        repository = repository_factory()
        _require_project(repository, project_id)
        try:
            note = repository.create_note(
                project_id,
                scope_kind=str((payload or {}).get("scope_kind") or "project"),
                source_id=(payload or {}).get("source_id"),
                evidence_id=(payload or {}).get("evidence_id"),
                title=str((payload or {}).get("title") or ""),
                content_markdown=str((payload or {}).get("content_markdown") or ""),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _note_dict(note)

    @app.patch("/api/v1/projects/{project_id}/notes/{note_id}")
    def api_update_note(project_id: str, note_id: str, payload: dict):
        repository = repository_factory()
        _require_note(repository, project_id, note_id)
        expected_version = _required_int(payload, "expected_version")
        fields = {
            key: payload[key]
            for key in ("title", "content_markdown")
            if key in (payload or {})
        }
        try:
            note = repository.update_note(note_id, expected_version=expected_version, **fields)
        except RecordVersionConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return _note_dict(note)

    @app.post("/ui/projects/{project_id}/notes")
    async def ui_create_note(request: Request, project_id: str):
        form = await _validated_form(request)
        repository = repository_factory()
        source_id = form.get("scope_source") or None
        try:
            repository.create_note(
                project_id,
                scope_kind="source" if source_id else "project",
                source_id=source_id,
                title=form.get("title", ""),
                content_markdown=form.get("content_markdown", ""),
            )
        except KeyError as exc:
            return _render_error(
                templates, request, _exception_message(exc), status_code=404
            )
        except ValueError as exc:
            return _render_error(templates, request, str(exc), status_code=400)
        return _notes_response(templates, request, repository, project_id)

    @app.post("/ui/projects/{project_id}/notes/{note_id}")
    async def ui_update_note(request: Request, project_id: str, note_id: str):
        form = await _validated_form(request)
        repository = repository_factory()
        try:
            expected_version = _form_int(form, "expected_version", minimum=1)
            _require_note(repository, project_id, note_id)
            repository.update_note(
                note_id,
                expected_version=expected_version,
                title=form.get("title", ""),
                content_markdown=form.get("content_markdown", ""),
            )
        except KeyError as exc:
            return _render_error(
                templates, request, _exception_message(exc), status_code=404
            )
        except RecordVersionConflictError as exc:
            return _render_error(templates, request, str(exc), status_code=409)
        except ValueError as exc:
            return _render_error(templates, request, str(exc), status_code=400)
        return _notes_response(templates, request, repository, project_id)


def _workspace_context(repository, store, project_id: str) -> dict[str, Any]:
    project = repository.get_project(project_id)
    if project is None:
        raise KeyError(f"研究项目不存在：{project_id}")
    questions = repository.list_questions(project_id)
    memberships = repository.list_project_sources(
        project_id, limit=200, offset=0
    ).items
    selected_paper_ids = {
        membership.source.indexed_paper_id
        for membership in memberships
        if membership.source.indexed_paper_id is not None
    }
    _, paper_rows = store.list_papers_with_chunk_counts(None, 200, 0)
    available_papers = [
        _paper_dict(paper, chunk_count, selected=paper.id in selected_paper_ids)
        for paper, chunk_count in paper_rows
        if paper.id not in selected_paper_ids
    ]
    return {
        "project": project,
        "questions": questions,
        "memberships": memberships,
        "available_papers": available_papers,
    }


def _question_response(templates, request, repository, project_id):
    if not _is_htmx(request):
        return RedirectResponse(f"/ui/projects/{project_id}", status_code=303)
    return _render(
        templates,
        request,
        "fragments/questions.html",
        {
            "project": _require_project(repository, project_id),
            "questions": repository.list_questions(project_id),
        },
    )


def _sources_response(templates, request, repository, store, project_id):
    if not _is_htmx(request):
        return RedirectResponse(f"/ui/projects/{project_id}", status_code=303)
    context = _workspace_context(repository, store, project_id)
    return _render(templates, request, "fragments/sources.html", context)


def _notes_response(templates, request, repository, project_id):
    if not _is_htmx(request):
        return RedirectResponse(
            f"/ui/projects/{project_id}/evidence", status_code=303
        )
    return _render(
        templates,
        request,
        "fragments/notes_panel.html",
        _notes_context(repository, project_id),
    )


def _notes_context(repository, project_id: str) -> dict[str, Any]:
    project = _require_project(repository, project_id)
    memberships = repository.list_project_sources(
        project_id, limit=200, offset=0
    ).items
    return {
        "project": project,
        "notes": repository.list_notes(project_id, limit=100, offset=0).items,
        "memberships": memberships,
    }


def _require_note(repository, project_id: str, note_id: str):
    note = repository.list_notes(project_id, limit=200, offset=0)
    for item in note.items:
        if item.id == note_id:
            return item
    raise KeyError("研究笔记不属于当前项目")


def _render(
    templates: Jinja2Templates,
    request: Request,
    template_name: str,
    context: dict[str, Any],
    *,
    status_code: int = 200,
):
    token, new_cookie = _csrf_token(request)
    values = {"request": request, "csrf_token": token, **context}
    response = templates.TemplateResponse(
        request=request,
        name=template_name,
        context=values,
        status_code=status_code,
    )
    if new_cookie:
        response.set_cookie(
            _CSRF_COOKIE,
            token,
            max_age=8 * 60 * 60,
            httponly=True,
            secure=request.url.scheme == "https",
            samesite="strict",
            path="/ui",
        )
    return response


def _render_error(
    templates: Jinja2Templates,
    request: Request,
    message: str,
    *,
    status_code: int,
):
    return _render(
        templates,
        request,
        "error.html",
        {"message": message},
        status_code=status_code,
    )


class _ValidatedForm(dict[str, str]):
    def __init__(self, parsed: dict[str, list[str]]) -> None:
        super().__init__({key: values[-1] for key, values in parsed.items() if values})
        self._parsed = parsed

    def getlist(self, key: str) -> list[str]:
        return list(self._parsed.get(key, ()))


async def _validated_form(request: Request) -> _ValidatedForm:
    content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if content_type != "application/x-www-form-urlencoded":
        raise HTTPException(status_code=415, detail="只接受 URL-encoded form")
    body = await request.body()
    if len(body) > _MAX_FORM_BYTES:
        raise HTTPException(status_code=413, detail="表单超过 1MB 限制")
    try:
        decoded = body.decode("utf-8", errors="strict")
        parsed = parse_qs(
            decoded,
            keep_blank_values=True,
            strict_parsing=False,
            max_num_fields=100,
        )
    except (UnicodeDecodeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="表单编码无效") from exc
    form = _ValidatedForm(parsed)
    cookie = request.cookies.get(_CSRF_COOKIE, "")
    supplied = form.get("csrf_token", "")
    if (
        not _valid_csrf_token(cookie)
        or not supplied
        or not secrets.compare_digest(cookie, supplied)
    ):
        raise HTTPException(status_code=403, detail="CSRF token 无效或已过期")
    return form


def _csrf_token(request: Request) -> tuple[str, bool]:
    current = request.cookies.get(_CSRF_COOKIE, "")
    if _valid_csrf_token(current):
        return current, False
    return secrets.token_urlsafe(32), True


def _valid_csrf_token(token: str) -> bool:
    return 32 <= len(token) <= 128 and all(
        character.isalnum() or character in {"-", "_"} for character in token
    )


def _is_htmx(request: Request) -> bool:
    return request.headers.get("HX-Request", "").lower() == "true"


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


def _form_int(form: dict[str, str], name: str, *, minimum: int) -> int:
    try:
        value = int(form.get(name, ""))
    except ValueError as exc:
        raise ValueError(f"{name} 必须是整数") from exc
    if value < minimum:
        raise ValueError(f"{name} 不能小于 {minimum}")
    return value


def _require_project(repository, project_id: str) -> ResearchProject:
    project = repository.get_project(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail=f"研究项目不存在：{project_id}")
    return project


def _project_dict(project: ResearchProject) -> dict[str, Any]:
    return {
        "id": project.id,
        "title": project.title,
        "description": project.description,
        "default_language": project.default_language,
        "citation_style": project.citation_style,
        "status": project.status,
        "version": project.version,
        "created_at": project.created_at,
        "updated_at": project.updated_at,
    }


def _question_dict(question: ResearchQuestion) -> dict[str, Any]:
    return {
        "id": question.id,
        "project_id": question.project_id,
        "question": question.question,
        "position": question.position,
        "version": question.version,
        "created_at": question.created_at,
        "updated_at": question.updated_at,
    }


def _membership_dict(membership: ProjectSourceMembership) -> dict[str, Any]:
    source = membership.source
    return {
        "project_id": membership.project_id,
        "position": membership.position,
        "note": membership.note,
        "added_at": membership.added_at,
        "source": {
            "id": source.id,
            "source_kind": source.source_kind,
            "title": source.title,
            "authors": list(source.authors),
            "year": source.year,
            "status": source.status,
            "indexed_paper_id": source.indexed_paper_id,
        },
    }


def _note_dict(note) -> dict[str, Any]:
    return {
        "id": note.id,
        "project_id": note.project_id,
        "scope_kind": note.scope_kind,
        "source_id": note.source_id,
        "evidence_id": note.evidence_id,
        "title": note.title,
        "content_markdown": note.content_markdown,
        "version": note.version,
        "created_at": note.created_at,
        "updated_at": note.updated_at,
    }


def _paper_dict(paper, chunk_count: int, *, selected: bool) -> dict[str, Any]:
    return {
        "id": paper.id,
        "title": paper.title,
        "authors": list(paper.authors),
        "year": paper.year,
        "page_count": paper.page_count,
        "chunk_count": chunk_count,
        "has_text": paper.has_text,
        "filename": Path(paper.path).name,
        "selected": selected,
    }


def _exception_message(exc: BaseException) -> str:
    return str(exc.args[0]) if getattr(exc, "args", None) else str(exc)
