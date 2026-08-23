"""pra 命令行入口。"""
import json as _json
import sys
from pathlib import Path

import typer

from . import __version__
from . import config
from .answer import answer_stream
from .answer import ask as answer_ask
from .embeddings import Embedder
from .import_pagent import ImportPagentError, import_pagent_data
from .indexer import index_library, validate_pdf_directory
from .llm import LLMClient, LLMError
from .search import hybrid_search
from .store import Store
from .websearch import WebSearchError, search_papers


def _configure_stream_utf8(stream) -> None:
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is None:
        return
    try:
        reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError, ValueError):
        # 测试捕获流或宿主包装流可能不允许重配置；这些流通常已支持 Unicode。
        pass


def _configure_stdio_utf8() -> None:
    """避免 Windows 的 GBK 标准流在输出论文 Unicode 文本时崩溃。"""
    for stream in (sys.stdout, sys.stderr):
        _configure_stream_utf8(stream)


_configure_stdio_utf8()

app = typer.Typer(help="PRAgent：论文整理与检索 Agent——索引本地 PDF 论文库，提供检索、问答与受控对话。")


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"pra {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="显示版本并退出",
    ),
):
    """PRAgent 论文整理与检索助手。"""


def _todo(cmd: str) -> None:
    typer.echo(f"[pra] {cmd} 尚未实现", err=True)
    raise typer.Exit(1)


@app.command("import-pagent")
def import_pagent_command(
    source: str = typer.Option("~/.pagent", "--source", help="旧 Pagent 数据目录"),
    target: str = typer.Option(None, "--target", help="目标目录；默认 PRA_DATA_DIR"),
    execute: bool = typer.Option(
        False,
        "--execute",
        help="实际执行；不提供时只做只读检查和 dry-run",
    ),
):
    """显式复制旧 Pagent 数据；默认 dry-run，绝不原地升级旧库。"""

    target_dir = Path(target).expanduser() if target else config.LIBRARY_DIR
    try:
        result = import_pagent_data(source, target_dir, execute=execute)
    except ImportPagentError as exc:
        typer.echo(f"错误：{exc}", err=True)
        raise typer.Exit(1)
    plan = result.plan
    typer.echo(
        f"旧库：schema v{plan.source_schema_version}，"
        f"论文 {plan.papers} 篇，分块 {plan.chunks} 条，"
        f"附加文件 {len(plan.files)} 个"
    )
    typer.echo(f"来源：{plan.source_dir}")
    typer.echo(f"目标：{plan.target_dir}")
    if plan.external_paper_paths:
        typer.echo(
            f"外部论文路径：{len(plan.external_paper_paths)} 个（验证后保留原引用，不复制）"
        )
    if result.executed:
        typer.echo(f"导入完成：目标 schema v{result.target_schema_version}")
    else:
        typer.echo("Dry-run 完成：未创建目标目录；确认后加 --execute 执行。")


@app.command()
def index(
    dir: str = typer.Argument(".", help="PDF 论文目录（默认当前目录）"),
    force: bool = typer.Option(False, "--force", help="显式全量替换索引（切换目录或嵌入模型时使用）"),
    refine: bool = typer.Option(False, "--refine", help="用 LLM 提炼元数据（需要配置 API key）"),
    no_prune: bool = typer.Option(False, "--no-prune", help="不删除库中已消失文件的条目"),
):
    """扫描目录中的 PDF 并建立索引。"""
    try:
        pdf_dir = validate_pdf_directory(Path(dir))
    except RuntimeError as exc:
        typer.echo(f"错误：{exc}", err=True)
        raise typer.Exit(1)
    config.ensure_data_dir()
    store = Store(config.DB_PATH)
    embedder = Embedder(config.EMBED_MODEL)
    llm = LLMClient(config.LLM_BASE_URL, config.LLM_API_KEY, config.LLM_MODEL) if refine else None
    try:
        result = index_library(
            store,
            pdf_dir,
            embedder,
            refine=refine,
            llm=llm,
            force=force,
            prune=not no_prune,
        )
    except RuntimeError as exc:
        typer.echo(f"错误：{exc}", err=True)
        raise typer.Exit(1)
    typer.echo(
        "完成："
        f"新增 {result['added']}，更新 {result['updated']}，未变化 {result['unchanged']}，"
        f"失败 {result['failed']}，删除 {result['removed']}，无文本 {result['skipped_no_text']}"
    )


@app.command()
def list(
    q: str = typer.Option(None, "--q", help="按标题/作者筛选"),
    json: bool = typer.Option(False, "--json", help="输出 JSON"),
):
    """列出库中论文。"""
    config.ensure_data_dir()
    store = Store(config.DB_PATH)
    total, papers = store.list_papers(q, 1000, 0)
    if json:
        typer.echo(
            _json.dumps(
                [
                    {
                        "id": p.id,
                        "title": p.title,
                        "authors": p.authors,
                        "year": p.year,
                        "path": p.path,
                        "page_count": p.page_count,
                        "has_text": p.has_text,
                    }
                    for p in papers
                ],
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if not papers:
        typer.echo("库为空。先运行 pra index <目录>。")
        return
    for p in papers:
        year = str(p.year) if p.year else "-"
        mark = "" if p.has_text else " [扫描版无文本]"
        typer.echo(f"{p.id}\t{p.title} ({year})\t{'、'.join(p.authors)}\t{p.path}{mark}")
    typer.echo(f"共 {total} 篇")


@app.command()
def status():
    """显示库与配置状态。"""
    config.ensure_data_dir()
    store = Store(config.DB_PATH)
    papers, chunks = store.stats()
    typer.echo(f"论文：{papers} 篇，分块：{chunks} 条")
    typer.echo(f"数据库目录：{config.LIBRARY_DIR}")
    typer.echo(f"嵌入模型：{store.meta_get('embed_model') or config.EMBED_MODEL}")
    typer.echo(f"论文目录：{store.meta_get('library_dir') or '（尚未索引）'}")
    if config.LLM_API_KEY:
        typer.echo(f"LLM：已配置（{config.LLM_MODEL} @ {config.LLM_BASE_URL}）")
    else:
        typer.echo("LLM：未配置（问答将退回纯检索，设置 PRA_LLM_API_KEY 启用生成式回答）")


@app.command()
def search(
    query: str = typer.Argument(..., help="检索词"),
    top: int = typer.Option(10, "--top", help="返回条数"),
    json: bool = typer.Option(False, "--json", help="输出 JSON"),
):
    """混合检索论文片段。"""
    config.ensure_data_dir()
    store = Store(config.DB_PATH)
    embedder = Embedder(config.EMBED_MODEL)
    hits = hybrid_search(store, embedder, query, top=top)
    if json:
        typer.echo(
            _json.dumps(
                [
                    {
                        "paper_id": h.paper_id,
                        "title": h.title,
                        "authors": h.authors,
                        "year": h.year,
                        "path": h.path,
                        "page": h.page,
                        "score": round(h.score, 6),
                        "text": h.text,
                    }
                    for h in hits
                ],
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if not hits:
        typer.echo("未找到相关内容。")
        return
    for h in hits:
        year = f"（{h.year}）" if h.year else ""
        typer.echo(f"[{h.score:.3f}] {h.title}{year} 第{h.page}页 — {h.path}")
        typer.echo(f"    {h.text.strip()[:120]}")
    typer.echo(f"共 {len(hits)} 条命中")


def _print_ask_sources(sources, web: bool, web_papers) -> None:
    """打印问答来源列表（流式与非流式路径共用）。"""
    if sources:
        typer.echo("\n来源：")
        for s in sources:
            year = f"（{s['year']}）" if s["year"] else ""
            if s.get("catalog"):
                tag = "（库藏）"
            else:
                tag = " [arXiv 联网]" if s["web"] else ""
            if s["page"] is not None:
                typer.echo(f"  [{s['n']}] {s['title']}{year} 第{s['page']}页 — {s['path']}{tag}")
            else:
                typer.echo(f"  [{s['n']}] {s['title']}{year} — {s['path']}{tag}")
    if web and not web_papers:
        typer.echo("\n（联网检索未找到相关结果，以上基于本地库回答；arXiv 以英文为主，英文问题更佳）")


def _print_web_papers(web_papers) -> None:
    if not web_papers:
        return
    typer.echo("\n联网检索（arXiv）：")
    for p in web_papers:
        year = f"（{p.year}）" if p.year else ""
        typer.echo(f"  {p.title}{year} — {p.url}" + (f" | PDF: {p.pdf_url}" if p.pdf_url else ""))


def _ask_stream_to_console(store, embedder, llm, question: str, top: int, web: bool) -> None:
    """流式问答：逐段打印回答，结束后展示来源；引用校验失败时给出警告。"""
    context = None
    verification = None
    for event in answer_stream(store, embedder, llm, question, top=top, web=web):
        if event["type"] == "context":
            context = event
        elif event["type"] == "delta":
            typer.echo(event["text"], nl=False)
            sys.stdout.flush()
        elif event["type"] == "complete":
            verification = event["verification"]
    typer.echo()
    if verification is not None and not verification["ok"]:
        typer.echo(
            f"警告：引用验证失败（{verification['code']}）：{verification['message']}", err=True
        )
    _print_ask_sources(context["sources"], web, context["web_papers"])


@app.command()
def ask(
    question: str = typer.Argument(..., help="问题"),
    top: int = typer.Option(8, "--top", help="检索块数"),
    no_llm: bool = typer.Option(False, "--no-llm", help="不调用 LLM，仅显示检索结果"),
    web: bool = typer.Option(False, "--web", help="同时联网检索 arXiv 论文（英文问题效果更佳）"),
    json: bool = typer.Option(False, "--json", help="输出 JSON"),
    no_stream: bool = typer.Option(False, "--no-stream", help="禁用流式输出，一次性返回完整回答"),
):
    """基于论文库回答问题（--web 时补充 arXiv 联网资料；默认流式输出）。"""
    config.ensure_data_dir()
    store = Store(config.DB_PATH)
    embedder = Embedder(config.EMBED_MODEL)
    llm = None if no_llm else LLMClient(config.LLM_BASE_URL, config.LLM_API_KEY, config.LLM_MODEL)
    use_stream = not json and not no_stream and llm is not None and llm.is_configured
    if use_stream:
        try:
            _ask_stream_to_console(store, embedder, llm, question, top=top, web=web)
            return
        except LLMError as exc:
            typer.echo(f"LLM 调用失败：{exc}（以下为检索结果）", err=True)
            _print_hits(hybrid_search(store, embedder, question, top=top, per_paper_cap=3))
            raise typer.Exit(1)
        except WebSearchError as exc:
            typer.echo(f"联网检索失败：{exc}", err=True)
            raise typer.Exit(1)
    try:
        answer_text, sources, hits, retrieval_only, web_papers = answer_ask(
            store, embedder, llm, question, top=top, web=web
        )
    except LLMError as exc:
        typer.echo(f"LLM 调用失败：{exc}（以下为检索结果）", err=True)
        hits = hybrid_search(store, embedder, question, top=top, per_paper_cap=3)
        _print_hits(hits)
        raise typer.Exit(1)
    except WebSearchError as exc:
        typer.echo(f"联网检索失败：{exc}", err=True)
        raise typer.Exit(1)

    if json:
        typer.echo(
            _json.dumps(
                {
                    "answer": answer_text,
                    "sources": sources,
                    "retrieval_only": retrieval_only,
                    "hits": [
                        {
                            "paper_id": h.paper_id,
                            "title": h.title,
                            "year": h.year,
                            "path": h.path,
                            "page": h.page,
                            "score": round(h.score, 6),
                            "text": h.text,
                        }
                        for h in hits
                    ],
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
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if answer_text is not None:
        typer.echo(answer_text)
        _print_ask_sources(sources, web, web_papers)
    else:
        if retrieval_only:
            typer.echo("未配置 PRA_LLM_API_KEY（或 --no-llm），仅显示检索结果；配置后获得生成式回答。")
        _print_hits(hits)
        _print_web_papers(web_papers)


@app.command()
def websearch(
    query: str = typer.Argument(..., help="检索词（英文效果最佳）"),
    top: int = typer.Option(5, "--top", help="返回条数"),
    json: bool = typer.Option(False, "--json", help="输出 JSON"),
):
    """联网检索 arXiv 论文。"""
    try:
        papers = search_papers(query, limit=top)
    except WebSearchError as exc:
        typer.echo(f"错误：{exc}", err=True)
        raise typer.Exit(1)
    if json:
        typer.echo(
            _json.dumps(
                [
                    {
                        "title": p.title,
                        "authors": p.authors,
                        "year": p.year,
                        "abstract": p.abstract,
                        "url": p.url,
                        "pdf_url": p.pdf_url,
                    }
                    for p in papers
                ],
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if not papers:
        typer.echo("未找到相关论文（arXiv 以英文为主，建议用英文查询）。")
        return
    for p in papers:
        year = f"（{p.year}）" if p.year else ""
        authors = "、".join(p.authors[:3]) + (" 等" if len(p.authors) > 3 else "")
        typer.echo(f"{p.title}{year} — {authors}")
        typer.echo(f"    {p.url}" + (f" | PDF: {p.pdf_url}" if p.pdf_url else ""))
        if p.abstract:
            typer.echo(f"    {p.abstract[:120]}")
    typer.echo(f"共 {len(papers)} 条结果")


def _print_hits(hits) -> None:
    if not hits:
        typer.echo("未找到相关内容。")
        return
    for h in hits:
        year = f"（{h.year}）" if h.year else ""
        typer.echo(f"[{h.score:.3f}] {h.title}{year} 第{h.page}页 — {h.path}")
        typer.echo(f"    {h.text.strip()[:120]}")
    typer.echo(f"共 {len(hits)} 条命中")


@app.command()
def chat():
    """TUI 对话模式：模型可自动调用工具（检索/搜索/下载/索引）。"""
    from .tools import ToolContext
    from .tui import ChatApp

    config.ensure_data_dir()
    llm = LLMClient(config.LLM_BASE_URL, config.LLM_API_KEY, config.LLM_MODEL)
    if not llm.is_configured:
        typer.echo("错误：未配置 PRA_LLM_API_KEY，对话模式需要 LLM。", err=True)
        raise typer.Exit(1)
    store = Store(config.DB_PATH)
    embedder = Embedder(config.EMBED_MODEL)
    ChatApp(llm=llm, ctx=ToolContext(store=store, embedder=embedder, llm=llm)).run()


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port"),
    ssl_certfile: str = typer.Option(None, "--ssl-certfile", help="HTTPS 证书文件"),
    ssl_keyfile: str = typer.Option(None, "--ssl-keyfile", help="HTTPS 私钥文件"),
    allow_insecure_http: bool = typer.Option(
        False,
        "--allow-insecure-http",
        help="仅可信网络：明确允许非本机明文 HTTP",
    ),
):
    """启动本地 Web 界面。"""
    from .webapp import serve as serve_web

    scheme = "https" if ssl_certfile and ssl_keyfile else "http"
    typer.echo(f"Web 界面：{scheme}://{host}:{port}（Ctrl+C 退出）")
    try:
        serve_web(
            host=host,
            port=port,
            ssl_certfile=ssl_certfile,
            ssl_keyfile=ssl_keyfile,
            allow_insecure_remote=allow_insecure_http,
        )
    except RuntimeError as exc:
        typer.echo(f"错误：{exc}", err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
