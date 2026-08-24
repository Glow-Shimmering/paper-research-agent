"""FastAPI Web 应用：检索 / 问答（含 SSE 流式）/ Agent（SSE 受控对话）/ 论文库 / 重新索引。"""
import json
from contextlib import asynccontextmanager
from importlib import resources
from pathlib import Path
import threading

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from . import config
from .agent_api import register_agent_api
from .answer import answer_stream
from .answer import ask as answer_ask
from .embeddings import Embedder
from .indexer import index_library
from .ingestion import FetchPolicy, SafeFetcher, SnapshotStore, WebIngestService
from .jobs import JobQueue, WorkerPool
from .llm import LLMClient, LLMError
from .research import DeepReadArtifactService, DeepReadWorkflow
from .search import hybrid_search
from .security import (
    api_key_matches,
    is_loopback_host,
    origin_matches_request,
    ui_auth_matches,
    ui_auth_token,
)
from .sources import (
    ArxivAdapter,
    CrossrefAdapter,
    DiscoveryService,
    SemanticScholarAdapter,
)
from .sources.actions import SourceActionService
from .storage import JobRepository, ResearchRepository
from .store import Store
from .web.routes import register_artifact_routes, register_project_routes
from .web.routes.discovery import register_discovery_routes
from .websearch import WebSearchError, search_papers


def _web_resource_directory(name: str) -> str:
    """返回随 wheel 安装的 Web 资源子目录。"""
    directory = resources.files("pragent").joinpath("web", name)
    if not directory.is_dir():
        raise RuntimeError(f"PRAgent 安装不完整：缺少 Web {name} 资源")
    return str(directory)


def _web_directory() -> str:
    """兼容工作台静态目录。"""
    return _web_resource_directory("legacy")


def _public_document_location(document) -> str:
    if getattr(document, "source_kind", "pdf") == "web":
        return getattr(document, "canonical_uri", None) or "网页来源"
    return Path(str(getattr(document, "path", ""))).name


def _hit_dict(h) -> dict:
    return {
        "paper_id": h.paper_id,
        "title": h.title,
        "authors": h.authors,
        "year": h.year,
        "path": _public_document_location(h),
        "source_kind": getattr(h, "source_kind", "pdf"),
        "canonical_uri": getattr(h, "canonical_uri", None),
        "page": h.page,
        "score": round(h.score, 6),
        "text": h.text,
    }


def _public_answer_source(source: dict) -> dict:
    result = dict(source)
    if not result.get("web") and "path" in result:
        result["path"] = Path(str(result["path"])).name
    return result


def _web_paper_dict(p) -> dict:
    return {
        "title": p.title,
        "authors": p.authors,
        "year": p.year,
        "abstract": p.abstract,
        "url": p.url,
        "pdf_url": p.pdf_url,
    }


def _sse_event(event: dict) -> dict:
    """把事件中的 dataclass 字段转换为 JSON 可序列化结构。"""
    data = dict(event)
    if data.get("type") == "context":
        data["hits"] = [_hit_dict(h) for h in data.get("hits", [])]
        data["sources"] = [
            _public_answer_source(source) for source in data.get("sources", [])
        ]
        data["web_papers"] = [_web_paper_dict(p) for p in data.get("web_papers", [])]
    return data


def create_app(
    store=None,
    embedder=None,
    llm=None,
    api_key: str | None = None,
    research_repository=None,
    source_providers=None,
    web_fetcher=None,
    snapshot_store=None,
    pdf_downloader=None,
    download_directory=None,
    job_repository=None,
    job_handlers=None,
    job_worker_count: int | None = None,
) -> FastAPI:
    """create_app(...) 可注入索引与研究 repository 依赖用于测试。"""
    owned_store: Store | None = None
    owned_research_repository: ResearchRepository | None = None
    owned_job_repository: JobRepository | None = None
    owned_worker_pool: WorkerPool | None = None
    owned_embedder = None
    owned_llm = None
    owned_source_providers = None
    owned_web_fetcher = None
    owned_snapshot_store = None
    dependency_lock = threading.RLock()
    reindex_lock = threading.Lock()
    network_slots = threading.BoundedSemaphore(4)
    expected_api_key = config.WEB_API_KEY if api_key is None else api_key

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        nonlocal owned_worker_pool
        queue = _job_queue()
        queue.recover_startup()
        handlers = _default_job_handlers()
        if job_handlers is not None:
            handlers.update(dict(job_handlers))
        owned_worker_pool = WorkerPool(
            queue,
            handlers,
            worker_count=(
                config.JOB_WORKERS if job_worker_count is None else job_worker_count
            ),
            poll_interval=config.JOB_POLL_SECONDS,
            lease_seconds=config.JOB_LEASE_SECONDS,
        )
        owned_worker_pool.start()
        _app.state.job_queue = queue
        _app.state.job_worker_pool = owned_worker_pool
        try:
            yield
        finally:
            owned_worker_pool.stop(grace_seconds=10)
            if owned_job_repository is not None:
                owned_job_repository.close()
            if owned_research_repository is not None:
                owned_research_repository.close()
            if owned_store is not None:
                owned_store.close()

    def _store() -> Store:
        nonlocal owned_store
        if store is not None:
            return store
        with dependency_lock:
            if owned_store is None:
                config.ensure_data_dir()
                owned_store = Store(config.DB_PATH)
            return owned_store

    def _research_repository() -> ResearchRepository:
        nonlocal owned_research_repository
        if research_repository is not None:
            return research_repository
        with dependency_lock:
            if owned_research_repository is None:
                db_path = _store().db_path
                if db_path is None:
                    raise RuntimeError(
                        "内存 Store 必须显式注入 ResearchRepository"
                    )
                owned_research_repository = ResearchRepository(db_path)
            return owned_research_repository

    def _job_queue() -> JobQueue:
        nonlocal owned_job_repository
        if job_repository is not None:
            return JobQueue(job_repository)
        with dependency_lock:
            if owned_job_repository is None:
                db_path = _store().db_path
                if db_path is None:
                    raise RuntimeError("内存 Store 必须显式注入 JobRepository")
                owned_job_repository = JobRepository(db_path)
            return JobQueue(owned_job_repository)

    def _default_job_handlers():
        def deep_read_handler(context, payload):
            context.report_progress(0, 9)
            workflow = DeepReadWorkflow(
                _store(),
                _embedder(),
                _llm(),
                on_progress=context.report_progress,
            )
            saved = DeepReadArtifactService(_research_repository()).generate_and_save(
                str(payload["project_id"]),
                str(payload["source_id"]),
                workflow,
                expected_artifact_version=int(payload["expected_artifact_version"]),
            )
            return {
                "artifact_id": saved.artifact.id,
                "revision_id": saved.revision.id,
            }

        def deep_read_field_handler(context, payload):
            context.report_progress(0, 1)
            workflow = DeepReadWorkflow(_store(), _embedder(), _llm())
            saved = DeepReadArtifactService(_research_repository()).regenerate_field(
                str(payload["project_id"]),
                str(payload["artifact_id"]),
                str(payload["field_name"]),
                workflow,
                expected_artifact_version=int(payload["expected_artifact_version"]),
                base_revision_id=str(payload["base_revision_id"]),
            )
            context.report_progress(1, 1)
            return {
                "artifact_id": saved.artifact.id,
                "revision_id": saved.revision.id,
            }

        return {
            "deep_read": deep_read_handler,
            "deep_read_field": deep_read_field_handler,
        }

    def _embedder():
        nonlocal owned_embedder
        if embedder is not None:
            return embedder
        with dependency_lock:
            if owned_embedder is None:
                owned_embedder = Embedder(config.EMBED_MODEL)
            return owned_embedder

    def _llm():
        nonlocal owned_llm
        if llm is not None:
            return llm
        with dependency_lock:
            if owned_llm is None:
                owned_llm = LLMClient(config.LLM_BASE_URL, config.LLM_API_KEY, config.LLM_MODEL)
            return owned_llm

    def _source_providers():
        nonlocal owned_source_providers
        if source_providers is not None:
            return tuple(source_providers)
        with dependency_lock:
            if owned_source_providers is None:
                owned_source_providers = (
                    ArxivAdapter(),
                    SemanticScholarAdapter(
                        api_key=config.SEMANTIC_SCHOLAR_API_KEY,
                        cache_directory=config.PROVIDER_CACHE_DIR,
                    ),
                    CrossrefAdapter(
                        contact_email=config.CROSSREF_EMAIL,
                        cache_directory=config.PROVIDER_CACHE_DIR,
                    ),
                )
            return owned_source_providers

    def _discovery_service():
        return DiscoveryService(
            _source_providers(), repository=_research_repository()
        )

    def _web_ingest_service():
        nonlocal owned_web_fetcher, owned_snapshot_store
        with dependency_lock:
            if web_fetcher is not None:
                fetcher = web_fetcher
            else:
                if owned_web_fetcher is None:
                    owned_web_fetcher = SafeFetcher(
                        policy=FetchPolicy(
                            max_redirects=config.WEB_FETCH_MAX_REDIRECTS,
                            timeout_seconds=config.WEB_FETCH_TIMEOUT_SECONDS,
                            max_response_bytes=config.WEB_FETCH_MAX_BYTES,
                        )
                    )
                fetcher = owned_web_fetcher
            if snapshot_store is not None:
                snapshots = snapshot_store
            else:
                if owned_snapshot_store is None:
                    owned_snapshot_store = SnapshotStore(
                        config.SNAPSHOT_DIR,
                        max_bytes=config.WEB_FETCH_MAX_BYTES,
                    )
                snapshots = owned_snapshot_store
        return WebIngestService(
            _research_repository(), fetcher=fetcher, snapshots=snapshots
        )

    def _action_service():
        target_directory = download_directory
        if target_directory is None:
            target_directory = config.download_dir_override()
        if target_directory is None:
            configured_library = _store().meta_get("library_dir")
            target_directory = (
                Path(configured_library)
                if configured_library
                else config.LIBRARY_DIR / "papers"
            )
        kwargs = {}
        if pdf_downloader is not None:
            kwargs["downloader"] = pdf_downloader
        return SourceActionService(
            _research_repository(),
            _store(),
            _embedder(),
            web_ingest=_web_ingest_service(),
            download_directory=target_directory,
            **kwargs,
        )

    app = FastAPI(title="PRAgent", lifespan=lifespan)

    @app.middleware("http")
    async def protect_api(request: Request, call_next):
        is_protected_ui = request.url.path.startswith("/ui/") and not request.url.path.startswith(
            "/ui/static/"
        )
        protected_path = request.url.path.startswith("/api/") or is_protected_ui
        if protected_path:
            request_host = request.url.hostname or ""
            if not expected_api_key and not is_loopback_host(request_host):
                return JSONResponse(
                    status_code=403,
                    content={"detail": "未配置 API key 时只接受 loopback Host"},
                )
            origin = request.headers.get("origin")
            if origin and not origin_matches_request(
                origin,
                request_scheme=request.url.scheme,
                request_host=request_host,
                request_port=request.url.port,
            ):
                return JSONResponse(status_code=403, content={"detail": "拒绝跨来源 API 请求"})
            raw_length = request.headers.get("content-length")
            if raw_length:
                try:
                    parsed_length = int(raw_length)
                    if parsed_length < 0:
                        raise ValueError
                    if parsed_length > 1_000_000:
                        return JSONResponse(status_code=413, content={"detail": "请求体超过 1MB 限制"})
                except ValueError:
                    return JSONResponse(status_code=400, content={"detail": "Content-Length 无效"})
            if expected_api_key:
                header_authenticated = api_key_matches(
                    request.headers.get("x-pra-key"), expected_api_key
                )
                ui_cookie_authenticated = is_protected_ui and ui_auth_matches(
                    request.cookies.get("pra_ui_auth"), expected_api_key
                )
                if not (header_authenticated or ui_cookie_authenticated):
                    return JSONResponse(
                        status_code=401,
                        content={"detail": "需要有效的 PRAgent API key"},
                    )
            if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
                chunks: list[bytes] = []
                received = 0
                async for chunk in request.stream():
                    received += len(chunk)
                    if received > 1_000_000:
                        return JSONResponse(
                            status_code=413,
                            content={"detail": "请求体超过 1MB 限制"},
                        )
                    chunks.append(chunk)
                # Starlette 会优先重放缓存的 _body，供 FastAPI 后续 JSON 解析。
                request._body = b"".join(chunks)
        return await call_next(request)

    @app.post("/api/ui-auth")
    def api_ui_auth(request: Request):
        """把已通过 X-PRA-Key 的浏览器切换为 HttpOnly 研究 UI cookie。"""

        response = JSONResponse({"ok": True})
        if expected_api_key:
            response.set_cookie(
                "pra_ui_auth",
                ui_auth_token(expected_api_key),
                max_age=8 * 60 * 60,
                httponly=True,
                secure=request.url.scheme == "https",
                samesite="strict",
                path="/ui",
            )
        return response

    @app.get("/api/status")
    def api_status():
        s = _store()
        papers, chunks = s.stats()
        return {
            "papers": papers,
            "chunks": chunks,
            "library_configured": bool(s.meta_get("library_dir")),
            "embed_model": s.meta_get("embed_model") or _embedder().model_name,
            "llm_configured": _llm().is_configured,
        }

    @app.get("/api/papers")
    def api_papers(
        q: str = Query("", max_length=500),
        limit: int = Query(50, ge=1, le=500),
        offset: int = Query(0, ge=0),
    ):
        s = _store()
        total, papers = s.list_papers_with_chunk_counts(q or None, limit, offset)
        return {
            "total": total,
            "items": [
                {
                    "id": p.id,
                    "title": p.title,
                    "authors": p.authors,
                    "year": p.year,
                    "path": _public_document_location(p),
                    "source_kind": p.source_kind,
                    "canonical_uri": p.canonical_uri,
                    "page_count": p.page_count,
                    "chunk_count": chunk_count,
                    "has_text": p.has_text,
                }
                for p, chunk_count in papers
            ],
        }

    @app.get("/api/search")
    def api_search(
        q: str = Query(..., min_length=1, max_length=2_000),
        top: int = Query(10, ge=1, le=100),
    ):
        hits = hybrid_search(_store(), _embedder(), q, top=top)
        return {"hits": [_hit_dict(h) for h in hits]}

    @app.post("/api/ask")
    def api_ask(payload: dict):
        question = (payload or {}).get("question", "")
        web = bool((payload or {}).get("web", False))
        if not question.strip():
            raise HTTPException(status_code=400, detail="question 不能为空")
        if len(question) > 20_000:
            raise HTTPException(status_code=413, detail="question 超过 20000 字符限制")
        if not network_slots.acquire(blocking=False):
            raise HTTPException(status_code=429, detail="联网/问答请求过多，请稍后重试")
        try:
            answer, sources, hits, retrieval_only, web_papers = answer_ask(
                _store(), _embedder(), _llm(), question, top=8, web=web
            )
        except LLMError as exc:
            raise HTTPException(status_code=500, detail=str(exc))
        except WebSearchError as exc:
            raise HTTPException(status_code=502, detail=str(exc))
        finally:
            network_slots.release()
        return {
            "answer": answer,
            "sources": [_public_answer_source(source) for source in sources],
            "retrieval_only": retrieval_only,
            "hits": [_hit_dict(h) for h in hits],
            "web_papers": [
                {
                    "title": p.title,
                    "authors": p.authors,
                    "year": p.year,
                    "abstract": p.abstract,
                    "url": p.url,
                    "pdf_url": p.pdf_url,
                }
                for p in web_papers
            ],
        }

    @app.post("/api/ask/stream")
    async def api_ask_stream(payload: dict):
        question = (payload or {}).get("question", "")
        web = bool((payload or {}).get("web", False))
        if not question.strip():
            raise HTTPException(status_code=400, detail="question 不能为空")
        if len(question) > 20_000:
            raise HTTPException(status_code=413, detail="question 超过 20000 字符限制")
        if not network_slots.acquire(blocking=False):
            raise HTTPException(status_code=429, detail="联网/问答请求过多，请稍后重试")
        try:
            gen = answer_stream(_store(), _embedder(), _llm(), question, top=8, web=web)
        except WebSearchError as exc:
            network_slots.release()
            raise HTTPException(status_code=502, detail=str(exc))
        except Exception:
            network_slots.release()
            raise

        async def sse_stream():
            """SSE 事件流：context → delta* → complete/error；断开时释放并发槽。"""
            sentinel = object()
            try:
                while True:
                    event = await run_in_threadpool(next, gen, sentinel)
                    if event is sentinel:
                        break
                    yield f"data: {json.dumps(_sse_event(event), ensure_ascii=False)}\n\n"
            except (LLMError, WebSearchError) as exc:
                payload = {"type": "error", "message": str(exc)}
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            finally:
                network_slots.release()

        return StreamingResponse(
            sse_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/websearch")
    def api_websearch(
        q: str = Query(..., min_length=1, max_length=500),
        top: int = Query(5, ge=1, le=10),
    ):
        if not q.strip():
            raise HTTPException(status_code=400, detail="q 不能为空")
        if not network_slots.acquire(blocking=False):
            raise HTTPException(status_code=429, detail="联网/问答请求过多，请稍后重试")
        try:
            papers = search_papers(q, limit=top)
        except WebSearchError as exc:
            raise HTTPException(status_code=502, detail=str(exc))
        finally:
            network_slots.release()
        return {
            "papers": [
                {
                    "title": p.title,
                    "authors": p.authors,
                    "year": p.year,
                    "abstract": p.abstract,
                    "url": p.url,
                    "pdf_url": p.pdf_url,
                }
                for p in papers
            ]
        }

    @app.post("/api/reindex")
    async def api_reindex():
        s = _store()
        lib_dir = s.meta_get("library_dir")
        if not lib_dir or not Path(lib_dir).is_dir():
            raise HTTPException(status_code=400, detail="尚未索引任何目录，请先运行 pra index <目录>")
        def reindex_locked():
            with reindex_lock:
                return index_library(
                    s,
                    Path(lib_dir),
                    _embedder(),
                    force=False,
                    prune=True,
                    progress=lambda msg: None,
                )

        return await run_in_threadpool(reindex_locked)

    register_agent_api(app, store_factory=_store, embedder_factory=_embedder, llm_factory=_llm)
    register_project_routes(
        app,
        store_factory=_store,
        repository_factory=_research_repository,
        templates_directory=_web_resource_directory("templates"),
    )
    register_artifact_routes(
        app,
        repository_factory=_research_repository,
        job_queue_factory=_job_queue,
        store_factory=_store,
        templates_directory=_web_resource_directory("templates"),
    )
    register_discovery_routes(
        app,
        repository_factory=_research_repository,
        discovery_service_factory=_discovery_service,
        action_service_factory=_action_service,
        templates_directory=_web_resource_directory("templates"),
    )
    app.mount(
        "/ui/static",
        StaticFiles(directory=_web_resource_directory("static")),
        name="research-static",
    )
    app.mount("/", StaticFiles(directory=_web_directory(), html=True), name="web")
    return app


def serve(
    host: str = "127.0.0.1",
    port: int = 8000,
    *,
    ssl_certfile: str | None = None,
    ssl_keyfile: str | None = None,
    allow_insecure_remote: bool = False,
) -> None:
    import uvicorn

    if bool(ssl_certfile) != bool(ssl_keyfile):
        raise RuntimeError("启用 HTTPS 时必须同时提供 --ssl-certfile 与 --ssl-keyfile")
    tls_options: dict[str, str] = {}
    if ssl_certfile and ssl_keyfile:
        for option, raw in (("ssl_certfile", ssl_certfile), ("ssl_keyfile", ssl_keyfile)):
            try:
                path = Path(raw).expanduser().resolve(strict=True)
            except (OSError, RuntimeError) as exc:
                raise RuntimeError(f"TLS 文件不存在或无法访问：{raw}") from exc
            if not path.is_file():
                raise RuntimeError(f"TLS 路径不是文件：{path}")
            tls_options[option] = str(path)

    if not is_loopback_host(host):
        if not config.WEB_API_KEY:
            raise RuntimeError(
                "拒绝无鉴权的非本机监听；请设置 PRA_WEB_API_KEY，或使用 --host 127.0.0.1"
            )
        if not tls_options and not allow_insecure_remote:
            raise RuntimeError(
                "拒绝通过明文 HTTP 远程传输 API key 和论文数据；请配置 TLS，"
                "使用 HTTPS 反向代理到 127.0.0.1，或仅在可信网络显式添加 --allow-insecure-http"
            )
    try:
        uvicorn.run(
            create_app(api_key=config.WEB_API_KEY),
            host=host,
            port=port,
            **tls_options,
        )
    except KeyboardInterrupt:
        # Windows/Python 3.11 下 uvicorn 会带着 KeyboardInterrupt 栈噪音退出；
        # 服务已正常关闭，静默吞掉避免吓到用户。
        pass
