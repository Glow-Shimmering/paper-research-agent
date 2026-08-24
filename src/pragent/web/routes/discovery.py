"""Phase 3 Discover/Library JSON API and HTMX routes."""

from __future__ import annotations

from typing import Any, Callable, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.concurrency import run_in_threadpool

from pragent.sources.actions import SourceActionError
from pragent.sources.identity import canonicalize_url
from pragent.storage import SourceIdentityConflictError

from .projects import (
    _exception_message,
    _form_int,
    _render,
    _render_error,
    _require_project,
    _validated_form,
)

_STATUS_LABELS = {
    "discovered": "已发现",
    "fetching": "处理中",
    "ready": "可用",
    "failed": "失败",
    "archived": "已归档",
}
_KIND_LABELS = {"paper": "论文", "web": "网页"}


def register_discovery_routes(
    app: FastAPI,
    *,
    repository_factory: Callable,
    discovery_service_factory: Callable,
    action_service_factory: Callable,
    templates_directory: str,
) -> None:
    templates = Jinja2Templates(directory=templates_directory)

    # ---------- JSON API ----------

    @app.get("/api/v1/sources")
    def api_sources(
        q: str = Query("", max_length=500),
        source_kind: Optional[str] = Query(None),
        status: Optional[str] = Query(None),
        limit: int = Query(50, ge=1, le=200),
        offset: int = Query(0, ge=0),
    ):
        try:
            repository = repository_factory()
            page = repository.list_sources(
                q=q or None,
                source_kind=source_kind,
                status=status,
                limit=limit,
                offset=offset,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "total": page.total,
            "limit": page.limit,
            "offset": page.offset,
            "items": [_public_source(repository, source) for source in page.items],
        }

    @app.post("/api/v1/discover/search")
    def api_discover_search(payload: dict):
        query = str((payload or {}).get("query") or "").strip()
        providers = _provider_list((payload or {}).get("providers"))
        limit = _strict_int((payload or {}).get("limit", 10), "limit", 1, 100)
        try:
            batch = discovery_service_factory().search(
                query,
                provider_names=providers,
                limit_per_provider=limit,
                persist=True,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _batch_dict(repository_factory(), batch)

    @app.post("/api/v1/sources/web", status_code=201)
    def api_import_web(payload: dict):
        url = str((payload or {}).get("url") or "").strip()
        project_id = _optional_text((payload or {}).get("project_id"))
        try:
            result = action_service_factory().import_web(
                url, project_id=project_id
            )
        except SourceActionError as exc:
            raise _action_http_error(exc) from exc
        return _action_dict(repository_factory(), result)

    @app.post("/api/v1/sources/{source_id}/download")
    def api_download_source(source_id: str, payload: Optional[dict] = None):
        project_id = _optional_text((payload or {}).get("project_id"))
        try:
            result = action_service_factory().download_and_index(
                source_id, project_id=project_id
            )
        except SourceActionError as exc:
            raise _action_http_error(exc) from exc
        return _action_dict(repository_factory(), result)

    @app.post("/api/v1/projects/{project_id}/sources/{source_id}", status_code=201)
    def api_select_discovered_source(project_id: str, source_id: str):
        repository = repository_factory()
        _require_project(repository, project_id)
        if repository.get_source(source_id) is None:
            raise HTTPException(status_code=404, detail="研究来源不存在")
        try:
            membership = repository.add_project_source(project_id, source_id)
        except (KeyError, SourceIdentityConflictError) as exc:
            raise HTTPException(
                status_code=409, detail=_exception_message(exc)
            ) from exc
        return {
            "project_id": membership.project_id,
            "source": _public_source(repository, membership.source),
            "position": membership.position,
            "added_at": membership.added_at,
        }

    # ---------- HTMX UI ----------

    @app.get("/ui/discover", response_class=HTMLResponse)
    def ui_discover(request: Request):
        repository = repository_factory()
        context = {
            "projects": repository.list_projects(limit=200, offset=0).items,
            "provider_names": tuple(discovery_service_factory().providers),
            "batch": None,
            "query": "",
            "selected_project_id": "",
        }
        return _render(templates, request, "discover.html", context)

    @app.post("/ui/discover/search")
    async def ui_discover_search(request: Request):
        form = await _validated_form(request)
        query = form.get("query", "").strip()
        providers = sorted(
            key.removeprefix("provider_")
            for key, value in form.items()
            if key.startswith("provider_") and value == "on"
        )
        project_id = form.get("project_id", "").strip()
        try:
            limit = _form_int(form, "limit", minimum=1)
            batch = await run_in_threadpool(
                discovery_service_factory().search,
                query,
                provider_names=providers,
                limit_per_provider=min(limit, 100),
                persist=True,
            )
            if project_id and repository_factory().get_project(project_id) is None:
                raise KeyError(f"研究项目不存在：{project_id}")
        except (ValueError, KeyError) as exc:
            return _render_error(
                templates, request, _exception_message(exc), status_code=400
            )
        return _render(
            templates,
            request,
            "fragments/discovery_results.html",
            {
                "results": _batch_dict(repository_factory(), batch),
                "query": query,
                "selected_project_id": project_id,
            },
        )

    @app.post("/ui/discover/web")
    async def ui_import_web(request: Request):
        form = await _validated_form(request)
        project_id = form.get("project_id", "").strip() or None
        try:
            result = await run_in_threadpool(
                action_service_factory().import_web,
                form.get("url", ""),
                project_id=project_id,
            )
        except SourceActionError as exc:
            return _render_action_error(templates, request, exc)
        return _render_action_result(
            templates, request, repository_factory(), result, "网页已抓取并索引"
        )

    @app.post("/ui/sources/{source_id}/download")
    async def ui_download_source(request: Request, source_id: str):
        form = await _validated_form(request)
        project_id = form.get("project_id", "").strip() or None
        try:
            result = await run_in_threadpool(
                action_service_factory().download_and_index,
                source_id,
                project_id=project_id,
            )
        except SourceActionError as exc:
            return _render_action_error(templates, request, exc)
        return _render_action_result(
            templates, request, repository_factory(), result, "PDF 已下载并索引"
        )

    @app.post("/ui/projects/{project_id}/sources/{source_id}")
    async def ui_select_source(request: Request, project_id: str, source_id: str):
        await _validated_form(request)
        repository = repository_factory()
        try:
            membership = repository.add_project_source(project_id, source_id)
        except KeyError as exc:
            return _render_error(
                templates, request, _exception_message(exc), status_code=404
            )
        return _render(
            templates,
            request,
            "fragments/action_result.html",
            {
                "ok": True,
                "message": "来源已加入研究项目",
                "source": _public_source(repository, membership.source),
            },
        )

    @app.get("/ui/library", response_class=HTMLResponse)
    def ui_library(
        request: Request,
        q: str = Query("", max_length=500),
        source_kind: Optional[str] = Query(None),
        status: Optional[str] = Query(None),
        offset: int = Query(0, ge=0),
    ):
        repository = repository_factory()
        try:
            page = repository.list_sources(
                q=q or None,
                source_kind=source_kind,
                status=status,
                limit=50,
                offset=offset,
            )
        except ValueError as exc:
            return _render_error(templates, request, str(exc), status_code=400)
        return _render(
            templates,
            request,
            "library.html",
            {
                "page": page,
                "sources": [
                    _public_source(repository, source) for source in page.items
                ],
                "projects": repository.list_projects(limit=200, offset=0).items,
                "q": q,
                "source_kind": source_kind or "",
                "status": status or "",
            },
        )

    @app.post("/ui/library/{source_id}/retry")
    async def ui_retry_source(request: Request, source_id: str):
        form = await _validated_form(request)
        repository = repository_factory()
        source = repository.get_source(source_id)
        if source is None:
            return _render_error(
                templates, request, "研究来源不存在", status_code=404
            )
        project_id = form.get("project_id", "").strip() or None
        try:
            if source.source_kind == "web" and source.canonical_url:
                result = await run_in_threadpool(
                    action_service_factory().import_web,
                    source.canonical_url,
                    project_id=project_id,
                )
            else:
                result = await run_in_threadpool(
                    action_service_factory().download_and_index,
                    source.id,
                    project_id=project_id,
                )
        except SourceActionError as exc:
            return _render_action_error(templates, request, exc)
        return _render_action_result(
            templates, request, repository, result, "来源已恢复并完成索引"
        )


def _public_source(repository, source) -> dict[str, Any]:
    providers = sorted(
        {record.provider for record in repository.list_source_records(source.id)}
    )
    last_error = source.metadata.get("last_error") if isinstance(source.metadata, dict) else None
    error_code = (
        str(last_error.get("code"))
        if isinstance(last_error, dict) and last_error.get("code")
        else None
    )
    try:
        canonical_url = (
            canonicalize_url(source.canonical_url)
            if source.canonical_url
            else None
        )
    except ValueError:
        canonical_url = None
    return {
        "id": source.id,
        "source_kind": source.source_kind,
        "source_kind_label": _KIND_LABELS.get(source.source_kind, source.source_kind),
        "title": source.title,
        "authors": list(source.authors),
        "year": source.year,
        "doi": source.doi,
        "arxiv_id": source.arxiv_id,
        "canonical_url": canonical_url,
        "status": source.status,
        "status_label": _STATUS_LABELS.get(source.status, source.status),
        "indexed": source.indexed_paper_id is not None,
        "indexed_paper_id": source.indexed_paper_id,
        "providers": providers,
        "can_download": bool(source.source_kind == "paper" and source.arxiv_id),
        "can_fetch": bool(source.source_kind == "web" and source.canonical_url),
        "error_code": error_code,
        "version": source.version,
        "created_at": source.created_at,
        "updated_at": source.updated_at,
    }


def _batch_dict(repository, batch) -> dict[str, Any]:
    return {
        "items": [
            {
                "source": _public_source(repository, item.persisted),
                "providers": list(item.merged.providers),
                "duplicate_count": item.merged.duplicate_count,
                "identities": [
                    {"kind": kind, "value": value}
                    for kind, value in item.merged.identities
                    if kind != "content_sha256"
                ],
            }
            for item in batch.items
            if item.persisted is not None
        ],
        "failures": [
            {
                "provider": failure.provider,
                "message": failure.message,
                "code": failure.code,
                "retryable": failure.retryable,
            }
            for failure in batch.failures
        ],
        "provider_counts": dict(batch.provider_counts),
    }


def _action_dict(repository, result) -> dict[str, Any]:
    return {
        "source": _public_source(repository, result.source),
        "paper_id": result.indexed.paper.id,
        "index_result": result.indexed.index_result,
        "deduplicated": result.indexed.deduplicated,
        "project_id": result.membership.project_id if result.membership else None,
    }


def _render_action_result(templates, request, repository, result, message):
    return _render(
        templates,
        request,
        "fragments/action_result.html",
        {
            "ok": True,
            "message": message,
            "source": _public_source(repository, result.source),
        },
    )


def _render_action_error(templates, request, error: SourceActionError):
    return _render(
        templates,
        request,
        "fragments/action_result.html",
        {
            "ok": False,
            "message": str(error),
            "error_code": error.code,
            "source": None,
        },
        status_code=502 if error.retryable else 409,
    )


def _action_http_error(error: SourceActionError) -> HTTPException:
    if error.code in {"source_not_found", "project_not_found"}:
        status_code = 404
    elif error.retryable:
        status_code = 502
    else:
        status_code = 409
    return HTTPException(
        status_code=status_code,
        detail={
            "code": error.code,
            "message": str(error),
            "retryable": error.retryable,
        },
    )


def _provider_list(value: Any) -> Optional[list[str]]:
    if value is None:
        return None
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise HTTPException(status_code=400, detail="providers 必须是字符串列表")
    return [item.strip().lower() for item in value if item.strip()]


def _strict_int(value: Any, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise HTTPException(status_code=400, detail=f"{name} 必须是整数")
    if not minimum <= value <= maximum:
        raise HTTPException(
            status_code=400,
            detail=f"{name} 必须在 {minimum}–{maximum} 之间",
        )
    return value


def _optional_text(value: Any) -> Optional[str]:
    text = str(value or "").strip()
    return text or None
