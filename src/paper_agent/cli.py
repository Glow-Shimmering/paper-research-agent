"""paper 命令行入口。"""
import json as _json
from pathlib import Path

import typer

from . import config
from .answer import ask as answer_ask
from .embeddings import Embedder
from .indexer import index_library
from .llm import LLMClient, LLMError
from .search import hybrid_search
from .store import Store
from .websearch import WebSearchError, search_papers

app = typer.Typer(help="论文整理与检索助手：索引本地 PDF 论文库，提供检索与问答。")


def _todo(cmd: str) -> None:
    typer.echo(f"[paper] {cmd} 尚未实现", err=True)
    raise typer.Exit(1)


@app.command()
def index(
    dir: str = typer.Argument(".", help="PDF 论文目录（默认当前目录）"),
    force: bool = typer.Option(False, "--force", help="嵌入模型变更时强制全量重建"),
    refine: bool = typer.Option(False, "--refine", help="用 LLM 提炼元数据（需要配置 API key）"),
    no_prune: bool = typer.Option(False, "--no-prune", help="不删除库中已消失文件的条目"),
):
    """扫描目录中的 PDF 并建立索引。"""
    config.ensure_data_dir()
    store = Store(config.DB_PATH)
    embedder = Embedder(config.EMBED_MODEL)
    llm = LLMClient(config.LLM_BASE_URL, config.LLM_API_KEY, config.LLM_MODEL) if refine else None
    try:
        result = index_library(
            store,
            Path(dir),
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
        typer.echo("库为空。先运行 paper index <目录>。")
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
        typer.echo("LLM：未配置（问答将退回纯检索，设置 PAPER_LLM_API_KEY 启用生成式回答）")


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


@app.command()
def ask(
    question: str = typer.Argument(..., help="问题"),
    top: int = typer.Option(8, "--top", help="检索块数"),
    no_llm: bool = typer.Option(False, "--no-llm", help="不调用 LLM，仅显示检索结果"),
    web: bool = typer.Option(False, "--web", help="同时联网检索 arXiv 论文（英文问题效果更佳）"),
    json: bool = typer.Option(False, "--json", help="输出 JSON"),
):
    """基于论文库回答问题（--web 时补充 arXiv 联网资料）。"""
    config.ensure_data_dir()
    store = Store(config.DB_PATH)
    embedder = Embedder(config.EMBED_MODEL)
    llm = None if no_llm else LLMClient(config.LLM_BASE_URL, config.LLM_API_KEY, config.LLM_MODEL)
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
        if sources:
            typer.echo("\n来源：")
            for s in sources:
                year = f"（{s['year']}）" if s["year"] else ""
                tag = " [arXiv 联网]" if s["web"] else ""
                if s["page"] is not None:
                    typer.echo(f"  [{s['n']}] {s['title']}{year} 第{s['page']}页 — {s['path']}{tag}")
                else:
                    typer.echo(f"  [{s['n']}] {s['title']}{year} — {s['path']}{tag}")
        if web and not web_papers:
            typer.echo("\n（联网检索未找到相关结果，以上基于本地库回答；arXiv 以英文为主，英文问题更佳）")
    else:
        if retrieval_only:
            typer.echo("未配置 PAPER_LLM_API_KEY（或 --no-llm），仅显示检索结果；配置后获得生成式回答。")
        _print_hits(hits)
        if web_papers:
            typer.echo("\n联网检索（arXiv）：")
            for p in web_papers:
                year = f"（{p.year}）" if p.year else ""
                typer.echo(f"  {p.title}{year} — {p.url}" + (f" | PDF: {p.pdf_url}" if p.pdf_url else ""))


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
def serve(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port"),
):
    """启动本地 Web 界面。"""
    from .webapp import serve as serve_web

    typer.echo(f"Web 界面：http://{host}:{port}（Ctrl+C 退出）")
    serve_web(host=host, port=port)


if __name__ == "__main__":
    app()
