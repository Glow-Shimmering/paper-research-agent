"""FastAPI Web 应用：检索 / 问答 / 论文库 / 重新索引。"""
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from . import config
from .answer import ask as answer_ask
from .embeddings import Embedder
from .indexer import index_library
from .llm import LLMClient, LLMError
from .search import hybrid_search
from .store import Store
from .websearch import WebSearchError, search_papers

WEB_DIR = Path(__file__).resolve().parent.parent.parent / "web"


def _hit_dict(h) -> dict:
    return {
        "paper_id": h.paper_id,
        "title": h.title,
        "authors": h.authors,
        "year": h.year,
        "path": h.path,
        "page": h.page,
        "score": round(h.score, 6),
        "text": h.text,
    }


def create_app(store=None, embedder=None, llm=None) -> FastAPI:
    """create_app(store=..., embedder=..., llm=...) 可注入依赖用于测试。"""

    def _store() -> Store:
        if store is not None:
            return store
        config.ensure_data_dir()
        return Store(config.DB_PATH)

    def _embedder():
        if embedder is not None:
            return embedder
        return Embedder(config.EMBED_MODEL)

    def _llm():
        if llm is not None:
            return llm
        return LLMClient(config.LLM_BASE_URL, config.LLM_API_KEY, config.LLM_MODEL)

    app = FastAPI(title="paper-agent")

    @app.get("/api/status")
    def api_status():
        s = _store()
        papers, chunks = s.stats()
        return {
            "papers": papers,
            "chunks": chunks,
            "library_dir": s.meta_get("library_dir"),
            "embed_model": s.meta_get("embed_model") or _embedder().model_name,
            "llm_configured": _llm().is_configured,
        }

    @app.get("/api/papers")
    def api_papers(q: str = "", limit: int = 50, offset: int = 0):
        s = _store()
        total, papers = s.list_papers(q or None, limit, offset)
        return {
            "total": total,
            "items": [
                {
                    "id": p.id,
                    "title": p.title,
                    "authors": p.authors,
                    "year": p.year,
                    "path": p.path,
                    "page_count": p.page_count,
                    "chunk_count": len(s.get_chunks_by_paper(p.id)),
                    "has_text": p.has_text,
                }
                for p in papers
            ],
        }

    @app.get("/api/search")
    def api_search(q: str = "", top: int = 10):
        hits = hybrid_search(_store(), _embedder(), q, top=top)
        return {"hits": [_hit_dict(h) for h in hits]}

    @app.post("/api/ask")
    def api_ask(payload: dict):
        question = (payload or {}).get("question", "")
        web = bool((payload or {}).get("web", False))
        if not question.strip():
            raise HTTPException(status_code=400, detail="question 不能为空")
        try:
            answer, sources, hits, retrieval_only, web_papers = answer_ask(
                _store(), _embedder(), _llm(), question, top=8, web=web
            )
        except LLMError as exc:
            raise HTTPException(status_code=500, detail=str(exc))
        except WebSearchError as exc:
            raise HTTPException(status_code=502, detail=str(exc))
        return {
            "answer": answer,
            "sources": sources,
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

    @app.get("/api/websearch")
    def api_websearch(q: str = "", top: int = 5):
        if not q.strip():
            raise HTTPException(status_code=400, detail="q 不能为空")
        try:
            papers = search_papers(q, limit=top)
        except WebSearchError as exc:
            raise HTTPException(status_code=502, detail=str(exc))
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
            raise HTTPException(status_code=400, detail="尚未索引任何目录，请先运行 paper index <目录>")
        return await run_in_threadpool(
            index_library,
            s,
            Path(lib_dir),
            _embedder(),
            force=False,
            prune=True,
            progress=lambda msg: None,
        )

    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
    return app


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(create_app(), host=host, port=port)
