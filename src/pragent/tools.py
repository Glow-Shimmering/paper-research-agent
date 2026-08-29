"""工具集：function calling 的 JSON schema 定义与执行器。

兼容基础检索、下载、索引和笔记工具，并提供论文内检索、逐页阅读、
分块上下文与稳定证据管理。实现尽量复用现有业务模块。
"""
import copy
import json
import hashlib
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from .models import Paper, SearchHit
from .tool_protocol import (
    ConfirmedAction,
    PendingAction,
    ToolEffect,
    ToolInterrupted,
    ToolResult,
    ToolSpec,
    ToolValidationError,
    validate_tool_arguments,
)

logger = logging.getLogger(__name__)

_MAX_OUTLINE_PAGES = 100
_MAX_OUTLINE_CHARS = 24_000
_MAX_READ_PAGE_SPAN = 50
_MAX_EVIDENCE_IDS_PER_PAGE = 50


@dataclass
class ToolContext:
    """工具执行上下文：对话进程内共享的真实依赖。

    ``deadline``/``cancel_event`` 构成真实的工具执行预算：deadline 由执行器
    按 ``ToolSpec.timeout_seconds`` 在每次调用前安装（单调时钟），cancel_event
    由会话层在客户端断开时置位。I/O handler 必须把剩余预算传给网络调用，
    长循环必须调用 ``ensure_runnable``/``is_runnable`` 检查；执行器不做
    "future 超时但副作用线程继续跑"的伪超时。
    """

    store: Any
    embedder: Any
    llm: Any
    session_id: Optional[str] = None
    project_id: Optional[str] = None
    research_repository: Any = None
    require_confirmation: bool = True
    pending_action: Optional[PendingAction | tuple[str, dict]] = None
    last_confirmed_action: Optional[ConfirmedAction] = None
    cancel_event: Optional[Any] = None
    deadline: Optional[float] = None

    def library_dir(self) -> Optional[Path]:
        raw = self.store.meta_get("library_dir")
        if not raw:
            return None
        path = Path(raw)
        return path if path.is_dir() else None

    def cancel_requested(self) -> bool:
        return self.cancel_event is not None and self.cancel_event.is_set()

    def remaining_seconds(self, default: Optional[float] = None) -> Optional[float]:
        """剩余执行预算；未安装 deadline 时返回 ``default``。"""
        if self.deadline is None:
            return default
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            return 0.0
        if default is not None:
            return min(remaining, float(default))
        return remaining

    def check_deadline(self) -> None:
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise ToolInterrupted(
                "tool_deadline_exceeded", "工具执行超出截止时间预算，操作已停止"
            )

    def check_cancelled(self) -> None:
        if self.cancel_requested():
            raise ToolInterrupted("tool_cancelled", "工具收到取消信号，操作已停止")

    def ensure_runnable(self) -> None:
        """I/O handler 在发起昂贵调用前检查取消与截止时间。"""
        self.check_cancelled()
        self.check_deadline()

    def is_runnable(self) -> bool:
        """供长循环逐项检查的布尔版本；不抛异常。"""
        if self.cancel_requested():
            return False
        return self.deadline is None or time.monotonic() < self.deadline


def _value(obj: Any, *names: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        for name in names:
            if name in obj:
                return obj[name]
        return default
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def _evidence_id(evidence: Any) -> Optional[str]:
    value = _value(evidence, "evidence_id", "id")
    return str(value) if value is not None else None


def _public_document_path(value: Any, *, source_kind: str = "pdf", canonical_uri=None) -> str:
    if source_kind == "web" and canonical_uri:
        return str(canonical_uri)
    path = str(value or "")
    if path.startswith("pragent-web://"):
        return path
    return Path(path).name


def _evidence_to_dict(evidence: Any) -> dict[str, Any]:
    if evidence is None:
        return {}
    keys = (
        "evidence_id",
        "id",
        "chunk_id",
        "paper_id",
        "source_hash",
        "paper_sha256",
        "chunk_text_sha256",
        "title",
        "authors",
        "path",
        "page",
        "chunk_seq",
        "text",
        "annotation",
        "pinned_at",
        "stale",
        "stale_reason",
        "created_at",
        "updated_at",
    )
    data = {
        key: _value(evidence, key)
        for key in keys
        if _value(evidence, key) is not None
    }
    stable_id = _evidence_id(evidence)
    if stable_id is not None:
        data["evidence_id"] = stable_id
        data.pop("id", None)
    if "path" in data:
        data["path"] = _public_document_path(data["path"])
    return data


def _chunk_to_dict(ctx: ToolContext, chunk: Any, *, max_chars: int = 2_000) -> dict:
    chunk_id = _value(chunk, "chunk_id", "id")
    evidence = ctx.store.evidence_from_chunk(int(chunk_id)) if chunk_id is not None else None
    text = str(_value(chunk, "text", default=""))
    return {
        "chunk_id": chunk_id,
        "paper_id": _value(chunk, "paper_id"),
        "seq": _value(chunk, "seq"),
        "page": _value(chunk, "page"),
        "text": text[:max_chars],
        "truncated": len(text) > max_chars,
        "evidence_id": _evidence_id(evidence),
    }


def _chunk_preview(chunks: list[Any], max_chars: int) -> str:
    """在不拼接整页全文的前提下生成有硬上限的预览。"""
    parts: list[str] = []
    remaining = max_chars
    for chunk in chunks:
        if remaining <= 0:
            break
        separator = " " if parts else ""
        if separator:
            parts.append(separator)
            remaining -= 1
        if remaining <= 0:
            break
        text = str(_value(chunk, "text", default=""))
        parts.append(text[:remaining])
        remaining -= min(len(text), remaining)
    return "".join(parts)


def _hit_to_dict(ctx: ToolContext, h: SearchHit) -> dict:
    evidence = ctx.store.evidence_from_chunk(h.chunk_id)
    return {
        "chunk_id": h.chunk_id,
        "paper_id": h.paper_id,
        "evidence_id": _evidence_id(evidence),
        "title": h.title,
        "year": h.year,
        "path": _public_document_path(
            h.path,
            source_kind=getattr(h, "source_kind", "pdf"),
            canonical_uri=getattr(h, "canonical_uri", None),
        ),
        "page": h.page,
        "score": round(h.score, 4),
        "text": h.text[:300],
    }


def _paper_to_dict(p: Any) -> dict:
    return {
        "id": _value(p, "paper_id", "id"),
        "title": _value(p, "title"),
        "authors": _value(p, "authors", default=[]),
        "year": _value(p, "year"),
        "path": _public_document_path(
            _value(p, "path"),
            source_kind=str(_value(p, "source_kind", default="pdf")),
            canonical_uri=_value(p, "canonical_uri"),
        ),
        "page_count": _value(p, "page_count"),
        "has_text": _value(p, "has_text"),
    }


def _local_search(ctx: ToolContext, query: str, top: int = 5) -> ToolResult | str:
    from .search import hybrid_search

    hits = hybrid_search(ctx.store, ctx.embedder, query, top=max(1, min(top, 20)))
    if not hits:
        return "本地库未找到相关内容。"
    rows = [_hit_to_dict(ctx, hit) for hit in hits]
    evidence_ids = tuple(row["evidence_id"] for row in rows if row["evidence_id"])
    return ToolResult.success(data=rows, evidence_ids=evidence_ids)


def _web_search(ctx: ToolContext, query: str, top: int = 5) -> ToolResult:
    from .websearch import search_papers

    # 网络调用使用当前工具执行预算的剩余时间，而不是固定超时。
    ctx.ensure_runnable()
    try:
        papers = search_papers(
            query, limit=max(1, min(top, 10)), timeout=ctx.remaining_seconds()
        )
    except Exception as exc:
        return ToolResult.error(
            "web_search_failed",
            f"联网检索失败：{exc}",
            retryable=True,
        )
    if not papers:
        return ToolResult.success(
            message="未找到相关论文（arXiv 以英文为主，建议用英文查询）。"
        )
    return ToolResult.success(
        data=[
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
    )


def _download_paper(
    ctx: ToolContext,
    url: str,
    _confirmed_target_dir: Optional[str] = None,
) -> ToolResult:
    from . import config
    from .download import DownloadError, download_pdf
    from .indexer import index_pdf

    # 下载目录：显式配置（PRA_DOWNLOAD_DIR / PRA_DATA_DIR）优先，否则论文库目录
    target_dir = (
        Path(_confirmed_target_dir).expanduser().resolve()
        if _confirmed_target_dir
        else config.download_dir_override() or ctx.library_dir()
    )
    if target_dir is None:
        return ToolResult.error(
            "download_dir_missing",
            (
                "未配置下载目录：请在 .env 设置 PRA_DOWNLOAD_DIR（或 PRA_DATA_DIR），"
                "或先运行 pra index <论文目录> 建立论文库。"
            ),
        )
    ctx.ensure_runnable()
    target_dir.mkdir(parents=True, exist_ok=True)
    remaining = ctx.remaining_seconds(default=60)
    timeout = 60 if remaining is None else max(remaining, 0.001)
    try:
        # 下载 I/O 使用剩余执行预算；超时由 socket 层真实中断，而非事后假装。
        path = download_pdf(url, target_dir, timeout=timeout)
    except DownloadError:
        return ToolResult.error(
            "download_failed",
            "PDF 下载失败，请检查 URL、网络与大小限制",
            retryable=True,
        )
    try:
        result = index_pdf(
            ctx.store,
            path,
            ctx.embedder,
            set_library_dir_if_missing=True,
            progress=lambda msg: None,
        )
    except Exception:
        return ToolResult.error(
            "download_index_failed",
            f"已下载 {path.name}，但索引失败。",
            data={"path": path.name},
        )
    if result["failed"]:
        return ToolResult.error(
            "download_index_failed",
            f"已下载 {path.name}，但索引失败；原有索引未清理。",
            data={"path": path.name, "index_result": result},
        )
    return ToolResult.success(
        message=(
            f"已下载并索引：{path.name}（新增 {result['added']}，更新 {result['updated']}，"
            f"未变化 {result['unchanged']}）"
        ),
        data={"path": path.name, "index_result": result},
    )


def _index_papers(ctx: ToolContext, dir: Optional[str] = None) -> ToolResult:
    from .indexer import index_library, validate_pdf_directory

    library_root = ctx.library_dir()
    if library_root is None:
        return ToolResult.error(
            "library_missing",
            "尚未建立论文库；请先在终端运行 pra index <论文目录>。",
        )
    target = Path(dir) if dir else library_root
    try:
        target = validate_pdf_directory(target)
    except RuntimeError:
        return ToolResult.error("invalid_library_directory", "论文库目录无效或无法访问")
    if target != library_root.resolve():
        return ToolResult.error(
            "library_switch_refused",
            (
                "拒绝通过 Agent 切换论文库目录。"
                "如需切换，请在终端显式运行 pra index <目录> --force。"
            ),
        )
    try:
        # 长循环逐篇检查取消/预算；已处理部分以一致的增量提交保留。
        result = index_library(
            ctx.store,
            target,
            ctx.embedder,
            progress=lambda msg: None,
            should_continue=ctx.is_runnable,
        )
    except Exception:
        return ToolResult.error("index_failed", "索引失败，请在终端检查本地文件与日志")
    message = (
        f"索引完成：新增 {result['added']}，更新 {result['updated']}，"
        f"未变化 {result['unchanged']}，失败 {result['failed']}，"
        f"无文本 {result['skipped_no_text']}"
    )
    if result.get("cancelled"):
        return ToolResult.error(
            "tool_cancelled",
            f"索引在取消信号/预算耗尽后提前结束；已处理部分保持一致。{message}",
            data=result,
            retryable=True,
        )
    if result["failed"]:
        return ToolResult.error("index_partial_failure", message, data=result)
    return ToolResult.success(message=message, data=result)


def _list_papers(ctx: ToolContext, q: Optional[str] = None) -> str:
    total, papers = ctx.store.list_papers(q or None, 100, 0)
    if not papers:
        return "库为空（共 0 篇）。"
    body = json.dumps([_paper_to_dict(p) for p in papers], ensure_ascii=False)
    return f"共 {total} 篇：\n{body}"


def _library_status(ctx: ToolContext) -> str:
    papers, chunks = ctx.store.stats()
    library_configured = bool(ctx.store.meta_get("library_dir"))
    embed_model = ctx.store.meta_get("embed_model") or "（未索引）"
    return (
        f"论文 {papers} 篇，分块 {chunks} 条；"
        f"论文目录：{'已配置' if library_configured else '未索引'}；"
        f"嵌入模型：{embed_model}"
    )


def _sanitize_filename(name: str) -> str:
    """清洗为合法 Windows 文件名：去非法字符、控制字符，截断长度。"""
    import re

    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip(" .")
    stem, dot, ext = name.rpartition(".")
    if len(name) > 100:
        name = (stem[: 100 - len(ext) - 1] + dot + ext) if dot else name[:100]
    return name


def _save_note(ctx: ToolContext, filename: str, content: str) -> ToolResult:
    """保存笔记到 notes 目录：自动创建目录；同名自动加后缀不覆盖；仅允许单文件名。"""
    from . import config

    if len(content) > 1_000_000:
        return ToolResult.error("note_too_large", "内容过长（超过 1MB），请分段保存。")
    name = _sanitize_filename(Path(filename).name)  # 只取 basename，防路径穿越
    if not name:
        return ToolResult.error("invalid_note_filename", "文件名无效。")
    notes = config.notes_dir()
    notes.mkdir(parents=True, exist_ok=True)
    target = notes / name
    if target.exists():
        stem, ext = target.stem, target.suffix
        i = 1
        while (notes / f"{stem} ({i}){ext}").exists():
            i += 1
        target = notes / f"{stem} ({i}){ext}"
    target.write_text(content, encoding="utf-8")
    return ToolResult.success(
        message=f"已保存：{target.name}",
        data={"path": target.name},
    )


def _list_notes(ctx: ToolContext) -> str:
    from . import config

    notes = config.notes_dir()
    if not notes.is_dir():
        return "notes 目录不存在（还没有保存过笔记）。"
    files = sorted(p for p in notes.iterdir() if p.is_file())
    if not files:
        return "notes 目录为空。"
    return json.dumps(
        [{"name": f.name, "size": f.stat().st_size} for f in files], ensure_ascii=False
    )


def _search_within_paper(
    ctx: ToolContext,
    paper_id: int,
    query: str,
    top: int = 5,
) -> ToolResult:
    from .search import search_within_paper

    paper = ctx.store.paper_by_id(paper_id)
    if paper is None:
        return ToolResult.error("paper_not_found", f"未找到论文 paper_id={paper_id}")
    hits = search_within_paper(
        ctx.store,
        ctx.embedder,
        paper_id,
        query,
        top=top,
    )
    rows = [_hit_to_dict(ctx, hit) for hit in hits]
    evidence_ids = tuple(row["evidence_id"] for row in rows if row["evidence_id"])
    return ToolResult.success(
        data={"paper": _paper_to_dict(paper), "hits": rows},
        evidence_ids=evidence_ids,
    )


def _get_paper_outline(
    ctx: ToolContext,
    paper_id: int,
    preview_chars: int = 240,
) -> ToolResult:
    paper = ctx.store.paper_by_id(paper_id)
    if paper is None:
        return ToolResult.error("paper_not_found", f"未找到论文 paper_id={paper_id}")
    chunks = list(ctx.store.paper_chunks(paper_id))
    by_page: dict[int, list[Any]] = {}
    for chunk in chunks:
        by_page.setdefault(int(_value(chunk, "page", default=0)), []).append(chunk)
    pages = []
    all_evidence: list[str] = []
    page_groups = sorted(by_page.items())
    remaining_preview_chars = _MAX_OUTLINE_CHARS
    for page, page_chunks in page_groups:
        if len(pages) >= _MAX_OUTLINE_PAGES or remaining_preview_chars <= 0:
            break
        evidence_ids = []
        for chunk in page_chunks[:_MAX_EVIDENCE_IDS_PER_PAGE]:
            chunk_id = _value(chunk, "chunk_id", "id")
            if chunk_id is None:
                continue
            stable_id = _evidence_id(ctx.store.evidence_from_chunk(int(chunk_id)))
            if stable_id:
                evidence_ids.append(stable_id)
                all_evidence.append(stable_id)
        preview = _chunk_preview(
            page_chunks,
            min(preview_chars, remaining_preview_chars),
        )
        remaining_preview_chars -= len(preview)
        pages.append(
            {
                "page": page,
                "chunk_count": len(page_chunks),
                "preview": preview[:preview_chars],
                "evidence_ids": evidence_ids,
                "evidence_ids_truncated": (
                    len(page_chunks) > _MAX_EVIDENCE_IDS_PER_PAGE
                ),
            }
        )
    return ToolResult.success(
        data={
            "paper": _paper_to_dict(paper),
            "pages": pages,
            "total_chunk_pages": len(page_groups),
            "truncated": len(pages) < len(page_groups),
            "limits": {
                "max_pages": _MAX_OUTLINE_PAGES,
                "max_preview_chars": _MAX_OUTLINE_CHARS,
            },
        },
        evidence_ids=tuple(dict.fromkeys(all_evidence)),
    )


def _read_pages(
    ctx: ToolContext,
    paper_id: int,
    start_page: int,
    end_page: Optional[int] = None,
    max_chars: int = 12_000,
) -> ToolResult:
    from .pdf import extract_pdf

    paper = ctx.store.paper_by_id(paper_id)
    if paper is None:
        return ToolResult.error("paper_not_found", f"未找到论文 paper_id={paper_id}")
    final_page = start_page if end_page is None else end_page
    if final_page < start_page:
        return ToolResult.error("invalid_page_range", "end_page 不能小于 start_page")
    if final_page - start_page + 1 > _MAX_READ_PAGE_SPAN:
        return ToolResult.error(
            "page_range_too_large",
            f"单次最多读取 {_MAX_READ_PAGE_SPAN} 页，请拆分请求。",
        )
    paper_path = Path(str(_value(paper, "path")))
    try:
        current_sha256 = _sha256_file(paper_path)
    except OSError as exc:
        return ToolResult.error(
            "paper_source_unavailable",
            f"无法读取索引论文文件：{exc}。请确认文件可访问后重新运行 pra index。",
        )
    indexed_sha256 = str(_value(paper, "sha256", default=""))
    if current_sha256 != indexed_sha256:
        return ToolResult.error(
            "paper_source_changed",
            "论文文件已在索引后发生变化；为避免文本与证据引用不一致，请重新运行 pra index 后重试。",
        )
    try:
        pages, _ = extract_pdf(paper_path)
    except Exception as exc:
        return ToolResult.error("pdf_read_failed", f"读取 PDF 失败：{exc}")
    if start_page > len(pages) or final_page > len(pages):
        return ToolResult.error(
            "page_out_of_range",
            f"页码超出范围；该论文共 {len(pages)} 页",
        )

    chunks_by_page: dict[int, list[Any]] = {}
    for chunk in ctx.store.paper_chunks(paper_id):
        chunks_by_page.setdefault(int(_value(chunk, "page", default=0)), []).append(chunk)
    remaining = max_chars
    output_pages = []
    all_evidence: list[str] = []
    for page_number in range(start_page, final_page + 1):
        full_text = pages[page_number - 1]
        text = full_text[:remaining]
        remaining -= len(text)
        evidence_ids = []
        page_chunks = chunks_by_page.get(page_number, [])
        for chunk in page_chunks[:_MAX_EVIDENCE_IDS_PER_PAGE]:
            chunk_id = _value(chunk, "chunk_id", "id")
            if chunk_id is None:
                continue
            stable_id = _evidence_id(ctx.store.evidence_from_chunk(int(chunk_id)))
            if stable_id:
                evidence_ids.append(stable_id)
                all_evidence.append(stable_id)
        output_pages.append(
            {
                "page": page_number,
                "text": text,
                "truncated": len(text) < len(full_text),
                "evidence_ids": evidence_ids,
                "evidence_ids_truncated": (
                    len(page_chunks) > _MAX_EVIDENCE_IDS_PER_PAGE
                ),
            }
        )
        if remaining <= 0:
            break
    return ToolResult.success(
        data={
            "paper": _paper_to_dict(paper),
            "requested_range": [start_page, final_page],
            "pages": output_pages,
            "budget_exhausted": remaining <= 0,
        },
        evidence_ids=tuple(dict.fromkeys(all_evidence)),
    )


def _read_chunk_context(
    ctx: ToolContext,
    chunk_id: int,
    before: int = 2,
    after: int = 2,
) -> ToolResult:
    raw_context = ctx.store.chunk_context(chunk_id, before, after)
    chunks = raw_context.get("chunks", []) if isinstance(raw_context, Mapping) else raw_context
    chunks = list(chunks or [])
    if not chunks:
        return ToolResult.error("chunk_not_found", f"未找到 chunk_id={chunk_id}")
    rows = [_chunk_to_dict(ctx, chunk) for chunk in chunks]
    evidence_ids = tuple(row["evidence_id"] for row in rows if row["evidence_id"])
    return ToolResult.success(
        data={"center_chunk_id": chunk_id, "chunks": rows},
        evidence_ids=evidence_ids,
    )


def _pin_evidence(
    ctx: ToolContext,
    chunk_id: int,
    annotation: str = "",
) -> ToolResult:
    evidence = ctx.store.pin_evidence(chunk_id, annotation)
    data = _evidence_to_dict(evidence)
    stable_id = _evidence_id(evidence)
    if stable_id is None:
        return ToolResult.error("evidence_pin_failed", "证据已固定，但存储层未返回 evidence_id")
    return ToolResult.success(data=data, evidence_ids=(stable_id,))


def _get_evidence(ctx: ToolContext, evidence_id: str) -> ToolResult:
    evidence = ctx.store.get_evidence(evidence_id)
    if evidence is None:
        return ToolResult.error("evidence_not_found", f"未找到证据 {evidence_id}")
    if bool(_value(evidence, "stale", default=False)):
        return ToolResult.success(
            data=_evidence_to_dict(evidence),
            message=(
                "该证据已过期，仅返回审计快照，不能作为当前答案的可引用证据："
                f"{_value(evidence, 'stale_reason', default='来源已变化')}"
            ),
        )
    return ToolResult.success(
        data=_evidence_to_dict(evidence),
        evidence_ids=(evidence_id,),
    )


def _list_evidence(ctx: ToolContext, limit: int = 20) -> ToolResult:
    evidence_items = list(ctx.store.list_evidence(limit))
    data = [_evidence_to_dict(item) for item in evidence_items]
    evidence_ids = tuple(
        stable_id
        for item in evidence_items
        if not bool(_value(item, "stale", default=False))
        for stable_id in (_evidence_id(item),)
        if stable_id is not None
    )
    stale_count = sum(bool(_value(item, "stale", default=False)) for item in evidence_items)
    message = ""
    if stale_count:
        message = (
            f"其中 {stale_count} 条证据已过期，仅作为审计快照展示，"
            "未加入当前答案的可引用证据。"
        )
    return ToolResult.success(data=data, message=message, evidence_ids=evidence_ids)


def _project_repository(ctx: ToolContext) -> tuple[Optional[str], Any, Optional[ToolResult]]:
    project_id = str(ctx.project_id or "").strip()
    repository = ctx.research_repository
    if not project_id or repository is None:
        return None, None, ToolResult.error(
            "project_context_required",
            "当前 Agent session 未绑定研究项目，不能读取项目资料。",
        )
    return project_id, repository, None


def _list_project_sources(ctx: ToolContext, limit: int = 50) -> ToolResult:
    project_id, repository, error = _project_repository(ctx)
    if error is not None:
        return error
    page = repository.list_project_sources(project_id, limit=limit)
    data = []
    for membership in page.items:
        source = membership.source
        data.append(
            {
                "source_id": source.id,
                "source_kind": source.source_kind,
                "title": source.title,
                "authors": list(source.authors),
                "year": source.year,
                "doi": source.doi,
                "arxiv_id": source.arxiv_id,
                "canonical_url": source.canonical_url,
                "status": source.status,
                "position": membership.position,
                "note": membership.note,
            }
        )
    return ToolResult.success(data={"total": page.total, "items": data})


def _list_project_artifacts(ctx: ToolContext, limit: int = 50) -> ToolResult:
    project_id, repository, error = _project_repository(ctx)
    if error is not None:
        return error
    page = repository.list_artifacts(project_id, limit=limit)
    items = []
    for artifact in page.items:
        revision = repository.get_current_artifact_revision(artifact.id)
        freshness = repository.artifact_freshness(artifact.id)
        items.append(
            {
                "artifact_id": artifact.id,
                "source_id": artifact.source_id,
                "artifact_type": artifact.artifact_type,
                "title": artifact.title,
                "status": artifact.status,
                "current_revision_number": artifact.current_revision_number,
                "revision_id": revision.id if revision else None,
                "stale": freshness.stale,
                "stale_reason": freshness.reason,
                "model": revision.model if revision else None,
                "prompt_version": revision.prompt_version if revision else None,
                "updated_at": artifact.updated_at,
            }
        )
    return ToolResult.success(data={"total": page.total, "items": items})


def _list_project_evidence(ctx: ToolContext, limit: int = 50) -> ToolResult:
    project_id, repository, error = _project_repository(ctx)
    if error is not None:
        return error
    artifacts = repository.list_artifacts(project_id, limit=200).items
    linked_ids: list[str] = []
    seen: set[str] = set()
    for artifact in artifacts:
        revision = repository.get_current_artifact_revision(artifact.id)
        if revision is None:
            continue
        for link in repository.list_artifact_evidence(revision.id):
            if link.evidence_id not in seen:
                linked_ids.append(link.evidence_id)
                seen.add(link.evidence_id)
            if len(linked_ids) >= limit:
                break
        if len(linked_ids) >= limit:
            break
    evidence_items = [ctx.store.get_evidence(item_id) for item_id in linked_ids]
    evidence_items = [item for item in evidence_items if item is not None]
    data = [_evidence_to_dict(item) for item in evidence_items]
    current_ids = tuple(
        item_id
        for item in evidence_items
        if not bool(_value(item, "stale", default=False))
        for item_id in (_evidence_id(item),)
        if item_id is not None
    )
    return ToolResult.success(data=data, evidence_ids=current_ids)


def _sha256_file(path: Path) -> str:
    """计算磁盘论文的内容哈希，确保页面文本与索引证据属于同一版本。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


_RAW_TOOL_DECLARATIONS: dict[str, tuple[dict, Any]] = {
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
    "save_note": (
        {
            "type": "function",
            "function": {
                "name": "save_note",
                "description": "把内容保存为本地笔记文件（保存到 notes 目录，自动创建目录；同名文件自动加序号后缀，不会覆盖已有文件；filename 只能是文件名，不支持子目录）。整理文献笔记、总结、导出对话等场景使用。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filename": {"type": "string", "description": "文件名，如 注意力机制笔记.md"},
                        "content": {"type": "string", "description": "要保存的文本内容"},
                    },
                    "required": ["filename", "content"],
                },
            },
        },
        _save_note,
    ),
    "list_notes": (
        {
            "type": "function",
            "function": {
                "name": "list_notes",
                "description": "列出 notes 目录中已保存的笔记文件（文件名与大小）。",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        _list_notes,
    ),
}

_REGISTRY: dict[str, ToolSpec] = {}
TOOLS: list[dict[str, Any]] = []
SCHEMA_NAMES: frozenset[str] = frozenset()
MUTATING_TOOLS: frozenset[str] = frozenset()
EXTERNAL_TOOLS: frozenset[str] = frozenset()
CONFIRMATION_TOOLS: frozenset[str] = frozenset()


def _refresh_tool_exports() -> None:
    """从唯一的 ToolSpec 注册表派生兼容导出，避免分类清单漂移。"""
    global SCHEMA_NAMES, MUTATING_TOOLS, EXTERNAL_TOOLS, CONFIRMATION_TOOLS

    TOOLS[:] = [spec.openai_schema() for spec in _REGISTRY.values()]
    SCHEMA_NAMES = frozenset(_REGISTRY)
    MUTATING_TOOLS = frozenset(
        name for name, spec in _REGISTRY.items() if spec.mutating
    )
    EXTERNAL_TOOLS = frozenset(
        name for name, spec in _REGISTRY.items() if spec.external
    )
    CONFIRMATION_TOOLS = MUTATING_TOOLS | EXTERNAL_TOOLS


def register_tool(spec: ToolSpec) -> None:
    """注册显式分类的工具；未使用 ToolSpec 或 effects 为空时立即失败。"""
    if not isinstance(spec, ToolSpec):
        raise ToolValidationError("工具必须通过 ToolSpec 注册并显式声明 effects")
    if not spec.effects:
        raise ToolValidationError(f"工具 {spec.name} effects 不能为空")
    if spec.name in _REGISTRY:
        raise ToolValidationError(f"工具 {spec.name} 重复注册")
    _REGISTRY[spec.name] = spec
    _refresh_tool_exports()


def unregister_tool(name: str) -> None:
    """移除已注册工具（供测试注入/清理；内置工具不应被移除）。"""
    if name not in _REGISTRY:
        return
    del _REGISTRY[name]
    _refresh_tool_exports()


def _constrained_parameters(
    parameters: Mapping[str, Any],
    constraints: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    normalized = copy.deepcopy(dict(parameters))
    properties = normalized.setdefault("properties", {})
    for field_name, rules in constraints.items():
        properties[field_name].update(rules)
    return normalized


_LEGACY_EFFECTS: dict[str, frozenset[ToolEffect]] = {
    "local_search": frozenset({ToolEffect.READ_LOCAL}),
    "web_search": frozenset({ToolEffect.NETWORK}),
    "download_paper": frozenset({ToolEffect.NETWORK, ToolEffect.WRITE_LOCAL}),
    "index_papers": frozenset({ToolEffect.READ_LOCAL, ToolEffect.WRITE_LOCAL}),
    "list_papers": frozenset({ToolEffect.READ_LOCAL}),
    "library_status": frozenset({ToolEffect.READ_LOCAL}),
    "save_note": frozenset({ToolEffect.WRITE_LOCAL}),
    "list_notes": frozenset({ToolEffect.READ_LOCAL}),
}

_PARAMETER_CONSTRAINTS: dict[str, dict[str, dict[str, Any]]] = {
    "local_search": {
        "query": {"minLength": 1, "maxLength": 2_000},
        "top": {"minimum": 1, "maximum": 20},
    },
    "web_search": {
        "query": {"minLength": 1, "maxLength": 2_000},
        "top": {"minimum": 1, "maximum": 10},
    },
    "download_paper": {"url": {"minLength": 1, "maxLength": 2_000}},
    "index_papers": {"dir": {"minLength": 1, "maxLength": 32_767}},
    "list_papers": {"q": {"maxLength": 500}},
    "save_note": {
        "filename": {"minLength": 1, "maxLength": 255},
        "content": {"maxLength": 1_000_000},
    },
}

_TOOL_RUNTIME_METADATA: dict[str, tuple[float, bool]] = {
    "local_search": (30.0, True),
    "web_search": (30.0, True),
    "download_paper": (180.0, False),
    "index_papers": (900.0, True),
    "list_papers": (10.0, True),
    "library_status": (5.0, True),
    "save_note": (10.0, False),
    "list_notes": (5.0, True),
}


for _name, (_schema, _handler) in _RAW_TOOL_DECLARATIONS.items():
    _function_schema = _schema["function"]
    _timeout_seconds, _idempotent = _TOOL_RUNTIME_METADATA[_name]
    register_tool(
        ToolSpec(
            name=_name,
            description=_function_schema["description"],
            parameters=_constrained_parameters(
                _function_schema["parameters"],
                _PARAMETER_CONSTRAINTS.get(_name, {}),
            ),
            handler=_handler,
            effects=_LEGACY_EFFECTS[_name],
            timeout_seconds=_timeout_seconds,
            idempotent=_idempotent,
        )
    )


_DEEP_READING_SPECS = (
    ToolSpec(
        name="search_within_paper",
        description="只在指定论文内进行关键词与语义混合检索，返回可引用的证据 ID。",
        parameters={
            "type": "object",
            "properties": {
                "paper_id": {"type": "integer", "minimum": 1},
                "query": {"type": "string", "minLength": 1, "maxLength": 2_000},
                "top": {"type": "integer", "minimum": 1, "maximum": 20},
            },
            "required": ["paper_id", "query"],
        },
        handler=_search_within_paper,
        effects=frozenset({ToolEffect.READ_LOCAL}),
        timeout_seconds=30.0,
        idempotent=True,
    ),
    ToolSpec(
        name="get_paper_outline",
        description="按页概览指定论文的分块、文本预览和稳定证据 ID。",
        parameters={
            "type": "object",
            "properties": {
                "paper_id": {"type": "integer", "minimum": 1},
                "preview_chars": {
                    "type": "integer",
                    "minimum": 50,
                    "maximum": 1_000,
                },
            },
            "required": ["paper_id"],
        },
        handler=_get_paper_outline,
        effects=frozenset({ToolEffect.READ_LOCAL}),
        timeout_seconds=30.0,
        idempotent=True,
    ),
    ToolSpec(
        name="read_pages",
        description="从指定论文 PDF 读取连续页，并返回对应的稳定证据 ID。",
        parameters={
            "type": "object",
            "properties": {
                "paper_id": {"type": "integer", "minimum": 1},
                "start_page": {"type": "integer", "minimum": 1},
                "end_page": {"type": "integer", "minimum": 1},
                "max_chars": {
                    "type": "integer",
                    "minimum": 100,
                    "maximum": 50_000,
                },
            },
            "required": ["paper_id", "start_page"],
        },
        handler=_read_pages,
        effects=frozenset({ToolEffect.READ_LOCAL}),
        timeout_seconds=60.0,
        idempotent=True,
    ),
    ToolSpec(
        name="read_chunk_context",
        description="读取目标分块及前后相邻分块，返回每段的稳定证据 ID。",
        parameters={
            "type": "object",
            "properties": {
                "chunk_id": {"type": "integer", "minimum": 1},
                "before": {"type": "integer", "minimum": 0, "maximum": 10},
                "after": {"type": "integer", "minimum": 0, "maximum": 10},
            },
            "required": ["chunk_id"],
        },
        handler=_read_chunk_context,
        effects=frozenset({ToolEffect.READ_LOCAL}),
        timeout_seconds=15.0,
        idempotent=True,
    ),
    ToolSpec(
        name="pin_evidence",
        description="把指定分块固定为可复用证据，并可附加批注。",
        parameters={
            "type": "object",
            "properties": {
                "chunk_id": {"type": "integer", "minimum": 1},
                "annotation": {"type": "string", "maxLength": 5_000},
            },
            "required": ["chunk_id"],
        },
        handler=_pin_evidence,
        effects=frozenset({ToolEffect.READ_LOCAL, ToolEffect.WRITE_LOCAL}),
        timeout_seconds=10.0,
        idempotent=True,
    ),
    ToolSpec(
        name="get_evidence",
        description="按稳定 evidence_id 读取一条已固定证据。",
        parameters={
            "type": "object",
            "properties": {
                "evidence_id": {
                    "type": "string",
                    "minLength": 3,
                    "maxLength": 128,
                }
            },
            "required": ["evidence_id"],
        },
        handler=_get_evidence,
        effects=frozenset({ToolEffect.READ_LOCAL}),
        timeout_seconds=10.0,
        idempotent=True,
    ),
    ToolSpec(
        name="list_evidence",
        description="列出最近固定的证据及其稳定 evidence_id。",
        parameters={
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "minimum": 1, "maximum": 100}
            },
        },
        handler=_list_evidence,
        effects=frozenset({ToolEffect.READ_LOCAL}),
        timeout_seconds=10.0,
        idempotent=True,
    ),
)

_PROJECT_READING_SPECS = (
    ToolSpec(
        name="list_project_sources",
        description="列出当前 Agent session 所绑定研究项目的来源元数据。",
        parameters={
            "type": "object",
            "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 100}},
        },
        handler=_list_project_sources,
        effects=frozenset({ToolEffect.READ_LOCAL}),
        timeout_seconds=10.0,
        idempotent=True,
    ),
    ToolSpec(
        name="list_project_artifacts",
        description="列出当前研究项目的 artifact、当前 revision 与新鲜度。",
        parameters={
            "type": "object",
            "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 100}},
        },
        handler=_list_project_artifacts,
        effects=frozenset({ToolEffect.READ_LOCAL}),
        timeout_seconds=10.0,
        idempotent=True,
    ),
    ToolSpec(
        name="list_project_evidence",
        description="列出当前研究项目最新 artifact revision 所引用的证据。",
        parameters={
            "type": "object",
            "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 100}},
        },
        handler=_list_project_evidence,
        effects=frozenset({ToolEffect.READ_LOCAL}),
        timeout_seconds=10.0,
        idempotent=True,
    ),
)

for _spec in (*_DEEP_READING_SPECS, *_PROJECT_READING_SPECS):
    register_tool(_spec)


def _pending_args(name: str, args: dict, ctx: ToolContext) -> dict:
    """冻结会影响落盘位置的派生参数，使确认内容与实际执行绑定。"""
    pending = dict(args)
    if name == "download_paper":
        from . import config

        target_dir = config.download_dir_override() or ctx.library_dir()
        pending = {"url": pending.get("url")}
        if target_dir is not None:
            pending["_confirmed_target_dir"] = str(target_dir.expanduser().resolve())
    return pending


def _coerce_pending_action(ctx: ToolContext) -> Optional[PendingAction]:
    pending = ctx.pending_action
    if pending is None:
        return None
    if isinstance(pending, PendingAction):
        return pending
    if (
        isinstance(pending, tuple)
        and len(pending) == 2
        and isinstance(pending[0], str)
        and isinstance(pending[1], Mapping)
    ):
        converted = PendingAction.create(pending[0], pending[1])
        ctx.pending_action = converted
        return converted
    return None


def pending_action_description(ctx: ToolContext, *, include_local_paths: bool = True) -> str:
    """返回可安全完整核对的待确认摘要；大文本使用长度、预览和哈希绑定。"""
    pending = _coerce_pending_action(ctx)
    if pending is None:
        return "没有待确认的工具操作。"
    display: dict[str, Any] = {
        "_action_id": pending.action_id,
        "_digest": pending.digest,
    }
    if pending.tool_call_id is not None:
        display["_tool_call_id"] = pending.tool_call_id
    if pending.run_id is not None:
        display["_run_id"] = pending.run_id
    for key, value in pending.args.items():
        if key.startswith("_confirmed_") and not include_local_paths:
            display[key] = "（仅在本地确认界面显示）"
            continue
        if isinstance(value, str) and len(value) > 500:
            display[key] = {
                "chars": len(value),
                "sha256": hashlib.sha256(value.encode("utf-8")).hexdigest(),
                "preview": value[:200],
            }
        else:
            display[key] = value
    return f"{pending.name}：{json.dumps(display, ensure_ascii=False)}"


def _unclassified_result(name: str) -> ToolResult:
    return ToolResult.error(
        "tool_unclassified",
        f"工具 {name} 未通过 ToolSpec 显式声明 effects，已拒绝执行。",
    )


def _handler_result(value: Any) -> ToolResult:
    if isinstance(value, ToolResult):
        return value
    if isinstance(value, (dict, list)):
        return ToolResult.success(data=value)
    return ToolResult.success(message=str(value))


def _run_handler(
    spec: ToolSpec,
    args: Mapping[str, Any],
    ctx: ToolContext,
    *,
    run_id: Optional[str] = None,
) -> ToolResult:
    """执行 handler 并安装真实执行预算。

    deadline 按 ``spec.timeout_seconds`` 以单调时钟安装到 ctx；若外层已设
    置更紧的 deadline（如任务级预算），取两者中更早者。handler 通过
    ``ensure_runnable``/``is_runnable`` 与网络超时参数消费预算。没有线程池
    future 包装：超时不是"放弃等待"，而是 handler 主动停止。
    """
    previous_deadline = ctx.deadline
    spec_deadline = time.monotonic() + spec.timeout_seconds
    if previous_deadline is not None:
        ctx.deadline = min(spec_deadline, previous_deadline)
    else:
        ctx.deadline = spec_deadline
    try:
        return _handler_result(spec.handler(ctx, **dict(args)))
    except ToolInterrupted as exc:
        logger.warning(
            "tool handler interrupted",
            extra={
                "session_id": ctx.session_id,
                "run_id": run_id,
                "tool_name": spec.name,
                "code": exc.code,
            },
        )
        # 幂等工具可安全重试；非幂等工具副作用未知，不允许自动重试。
        return ToolResult.error(exc.code, str(exc), retryable=spec.idempotent)
    except Exception as exc:
        logger.exception(
            "tool handler failed",
            extra={
                "session_id": ctx.session_id,
                "run_id": run_id,
                "tool_name": spec.name,
                "error_type": type(exc).__name__,
            },
        )
        return ToolResult.error(
            "tool_execution_failed",
            f"工具 {spec.name} 执行失败，请检查本地日志",
            retryable=spec.external,
        )
    finally:
        ctx.deadline = previous_deadline


def _confirmation_result(pending: PendingAction, ctx: ToolContext) -> ToolResult:
    return ToolResult(
        ok=False,
        code="confirmation_required",
        message=(
            "操作尚未执行。需要用户确认外部/写入操作："
            f"{pending_action_description(ctx, include_local_paths=False)}。"
            "请停止继续调用需确认的操作，并请用户在 TUI 输入 /confirm；"
            "输入 /cancel 可取消。"
        ),
        requires_confirmation=True,
        action_id=pending.action_id,
        digest=pending.digest,
    )


def execute_tool_result(
    name: str,
    args: Any,
    ctx: ToolContext,
    *,
    confirmed: bool = False,
    tool_call_id: Optional[str] = None,
    run_id: Optional[str] = None,
    action_id: Optional[str] = None,
    digest: Optional[str] = None,
) -> ToolResult:
    """执行工具并返回结构化结果；确认时只执行已绑定的冻结参数。"""
    entry = _REGISTRY.get(name)
    if entry is None:
        return ToolResult.error(
            "unknown_tool",
            f"未知工具：{name}。可用工具：{', '.join(sorted(SCHEMA_NAMES))}",
        )
    if not isinstance(entry, ToolSpec) or not entry.effects:
        return _unclassified_result(name)

    if confirmed:
        pending = _coerce_pending_action(ctx)
        if pending is None:
            return ToolResult.error("confirmation_missing", "没有待确认的工具操作。")
        if action_id is None or digest is None:
            return ToolResult.error(
                "confirmation_binding_required",
                "确认执行必须同时提供 action_id 与 digest。",
            )
        if pending.name != name:
            return ToolResult.error(
                "confirmation_mismatch",
                f"确认票据属于 {pending.name}，不能执行 {name}。",
            )
        if action_id != pending.action_id:
            return ToolResult.error("confirmation_mismatch", "action_id 与待确认操作不匹配。")
        if digest != pending.digest:
            return ToolResult.error("confirmation_mismatch", "digest 与待确认参数不匹配。")
        if pending.tool_call_id is not None and tool_call_id != pending.tool_call_id:
            return ToolResult.error(
                "confirmation_mismatch",
                "tool_call_id 与待确认操作不匹配。",
            )
        if pending.run_id is not None and run_id != pending.run_id:
            return ToolResult.error(
                "confirmation_mismatch",
                "run_id 与待确认操作不匹配。",
            )
        if not pending.is_bound():
            ctx.pending_action = None
            return ToolResult.error(
                "confirmation_mismatch",
                "待确认参数已发生变化，操作已取消。",
            )
        ctx.pending_action = None
        result = _run_handler(
            entry,
            pending.args,
            ctx,
            run_id=pending.run_id or run_id,
        )
        ctx.last_confirmed_action = ConfirmedAction(
            name=pending.name,
            args=copy.deepcopy(pending.args),
            action_id=pending.action_id,
            digest=pending.digest,
            result=result,
            tool_call_id=pending.tool_call_id,
            run_id=pending.run_id,
        )
        return result

    try:
        normalized_args = validate_tool_arguments(entry, args)
    except ToolValidationError as exc:
        return ToolResult.error("invalid_arguments", f"工具参数错误：{exc}")

    if entry.needs_confirmation and ctx.require_confirmation:
        pending = _coerce_pending_action(ctx)
        if pending is None:
            pending = PendingAction.create(
                name,
                _pending_args(name, normalized_args, ctx),
                tool_call_id=tool_call_id,
                run_id=run_id,
            )
            ctx.pending_action = pending
        return _confirmation_result(pending, ctx)
    return _run_handler(entry, normalized_args, ctx, run_id=run_id)


def execute_tool(name: str, args: dict, ctx: ToolContext, *, confirmed: bool = False) -> str:
    """执行工具，返回给 LLM 的文本结果。未知工具返回错误说明。"""
    return execute_tool_result(name, args, ctx, confirmed=confirmed).to_model_text()


def confirm_pending_action(ctx: ToolContext) -> tuple[str, str]:
    """执行用户已确认的精确待办参数，返回 (tool_name, result)。"""
    pending = _coerce_pending_action(ctx)
    if pending is None:
        return "", "没有待确认的工具操作。"
    result = execute_tool_result(
        pending.name,
        pending.args,
        ctx,
        confirmed=True,
        tool_call_id=pending.tool_call_id,
        run_id=pending.run_id,
        action_id=pending.action_id,
        digest=pending.digest,
    )
    return pending.name, result.to_model_text()


def cancel_pending_action(ctx: ToolContext) -> bool:
    had_pending = ctx.pending_action is not None
    ctx.pending_action = None
    return had_pending
