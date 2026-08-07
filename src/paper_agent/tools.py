"""工具集：function calling 的 JSON schema 定义与执行器。

工具：local_search / web_search / download_paper / index_papers / list_papers / library_status。
实现里尽量复用现有模块（search / websearch / indexer / download / store）。
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from .models import Paper, SearchHit


@dataclass
class ToolContext:
    """工具执行上下文：对话进程内共享的真实依赖。"""

    store: Any
    embedder: Any
    llm: Any

    def library_dir(self) -> Optional[Path]:
        raw = self.store.meta_get("library_dir")
        if not raw:
            return None
        path = Path(raw)
        return path if path.is_dir() else None


def _hit_to_dict(h: SearchHit) -> dict:
    return {
        "title": h.title,
        "year": h.year,
        "path": h.path,
        "page": h.page,
        "score": round(h.score, 4),
        "text": h.text[:300],
    }


def _paper_to_dict(p: Paper) -> dict:
    return {
        "id": p.id,
        "title": p.title,
        "authors": p.authors,
        "year": p.year,
        "path": p.path,
        "page_count": p.page_count,
        "has_text": p.has_text,
    }


def _local_search(ctx: ToolContext, query: str, top: int = 5) -> str:
    from .search import hybrid_search

    hits = hybrid_search(ctx.store, ctx.embedder, query, top=max(1, min(top, 20)))
    if not hits:
        return "本地库未找到相关内容。"
    return json.dumps([_hit_to_dict(h) for h in hits], ensure_ascii=False)


def _web_search(ctx: ToolContext, query: str, top: int = 5) -> str:
    from .websearch import search_papers

    try:
        papers = search_papers(query, limit=max(1, min(top, 10)))
    except Exception as exc:
        return f"联网检索失败：{exc}"
    if not papers:
        return "未找到相关论文（arXiv 以英文为主，建议用英文查询）。"
    return json.dumps(
        [
            {
                "title": p.title,
                "authors": p.authors,
                "year": p.year,
                "abstract": p.abstract[:300],
                "url": p.url,
                "pdf_url": p.pdf_url,
            }
            for p in papers
        ],
        ensure_ascii=False,
    )


def _download_paper(ctx: ToolContext, url: str) -> str:
    from . import config
    from .download import DownloadError, download_pdf
    from .indexer import index_library

    # 下载目录：显式配置（PAPER_DOWNLOAD_DIR / PAPER_DATA_DIR）优先，否则论文库目录
    target_dir = config.download_dir_override() or ctx.library_dir()
    if target_dir is None:
        return (
            "未配置下载目录：请在 .env 设置 PAPER_DOWNLOAD_DIR（或 PAPER_DATA_DIR），"
            "或先运行 paper index <论文目录> 建立论文库。"
        )
    target_dir.mkdir(parents=True, exist_ok=True)
    try:
        path = download_pdf(url, target_dir)
    except DownloadError as exc:
        return f"下载失败：{exc}"
    try:
        result = index_library(ctx.store, target_dir, ctx.embedder, progress=lambda msg: None)
    except Exception as exc:
        return f"已下载到 {path}，但索引失败：{exc}"
    return (
        f"已下载并索引：{path}（新增 {result['added']}，更新 {result['updated']}，"
        f"未变化 {result['unchanged']}）"
    )


def _index_papers(ctx: ToolContext, dir: Optional[str] = None) -> str:
    from .indexer import index_library

    target = Path(dir) if dir else Path(ctx.store.meta_get("library_dir") or ".")
    if not target.is_dir():
        return f"目录不存在：{target}"
    try:
        result = index_library(ctx.store, target, ctx.embedder, progress=lambda msg: None)
    except Exception as exc:
        return f"索引失败：{exc}"
    return (
        f"索引完成：新增 {result['added']}，更新 {result['updated']}，"
        f"未变化 {result['unchanged']}，失败 {result['failed']}，无文本 {result['skipped_no_text']}"
    )


def _list_papers(ctx: ToolContext, q: Optional[str] = None) -> str:
    total, papers = ctx.store.list_papers(q or None, 100, 0)
    if not papers:
        return "库为空（共 0 篇）。"
    body = json.dumps([_paper_to_dict(p) for p in papers], ensure_ascii=False)
    return f"共 {total} 篇：\n{body}"


def _library_status(ctx: ToolContext) -> str:
    papers, chunks = ctx.store.stats()
    lib_dir = ctx.store.meta_get("library_dir") or "（未索引）"
    embed_model = ctx.store.meta_get("embed_model") or "（未索引）"
    return (
        f"论文 {papers} 篇，分块 {chunks} 条；"
        f"论文目录：{lib_dir}；嵌入模型：{embed_model}"
    )


_REGISTRY: dict[str, tuple[dict, Callable]] = {
    "local_search": (
        {
            "type": "function",
            "function": {
                "name": "local_search",
                "description": "在本地论文库中检索相关内容片段（关键词+语义混合）。回答与本地论文相关的问题前应先用它。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "检索词，中英文均可"},
                        "top": {"type": "integer", "description": "返回条数，默认 5"},
                    },
                    "required": ["query"],
                },
            },
        },
        _local_search,
    ),
    "web_search": (
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "联网检索 arXiv 论文（免费）。查找最新论文、本地库没有的资料时使用；英文关键词效果更佳。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "检索词，建议用英文"},
                        "top": {"type": "integer", "description": "返回条数，默认 5"},
                    },
                    "required": ["query"],
                },
            },
        },
        _web_search,
    ),
    "download_paper": (
        {
            "type": "function",
            "function": {
                "name": "download_paper",
                "description": "下载 arXiv 论文 PDF 到本地论文库目录并自动建立索引。参数为 arXiv 摘要页或 PDF 页 URL（如 https://arxiv.org/abs/2402.11651）。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "arXiv 论文 URL"},
                    },
                    "required": ["url"],
                },
            },
        },
        _download_paper,
    ),
    "index_papers": (
        {
            "type": "function",
            "function": {
                "name": "index_papers",
                "description": "扫描并索引 PDF 论文目录（增量，只处理变化的文件）。不传目录时使用库的论文目录。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dir": {"type": "string", "description": "论文目录路径，可选"},
                    },
                },
            },
        },
        _index_papers,
    ),
    "list_papers": (
        {
            "type": "function",
            "function": {
                "name": "list_papers",
                "description": "列出本地库中的论文（标题/作者/年份/路径）。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "q": {"type": "string", "description": "按标题/作者筛选词，可选"},
                    },
                },
            },
        },
        _list_papers,
    ),
    "library_status": (
        {
            "type": "function",
            "function": {
                "name": "library_status",
                "description": "查看本地论文库状态：论文数、分块数、论文目录、嵌入模型。",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        _library_status,
    ),
}

TOOLS: list[dict] = [schema for schema, _ in _REGISTRY.values()]

SCHEMA_NAMES = set(_REGISTRY.keys())


def execute_tool(name: str, args: dict, ctx: ToolContext) -> str:
    """执行工具，返回给 LLM 的文本结果。未知工具返回错误说明。"""
    entry = _REGISTRY.get(name)
    if entry is None:
        return f"未知工具：{name}。可用工具：{', '.join(sorted(SCHEMA_NAMES))}"
    impl = entry[1]
    try:
        return impl(ctx, **(args or {}))
    except TypeError as exc:
        return f"工具参数错误：{exc}"
    except Exception as exc:
        return f"工具执行失败：{exc}"
