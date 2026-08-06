"""索引管线：扫描 PDF 目录 → 解析 → 分块 → 嵌入 → 入库。"""
import hashlib
from pathlib import Path
from typing import Callable, Optional

from .chunking import chunk_text
from .llm import refine_metadata
from .models import Chunk, Paper
from .pdf import extract_pdf, guess_metadata
from .store import Store, now_iso


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def index_library(
    store: Store,
    pdf_dir: Path,
    embedder,
    *,
    refine: bool = False,
    llm=None,
    force: bool = False,
    prune: bool = True,
    progress: Callable[[str], None] = print,
) -> dict:
    """增量索引。返回 {added, updated, unchanged, failed, removed, skipped_no_text}。"""
    result = {
        "added": 0,
        "updated": 0,
        "unchanged": 0,
        "failed": 0,
        "removed": 0,
        "skipped_no_text": 0,
    }

    stored_model = store.meta_get("embed_model")
    if stored_model is not None and stored_model != embedder.model_name and not force:
        raise RuntimeError(
            f"库由嵌入模型「{stored_model}」建立，当前为「{embedder.model_name}」。"
            "如需切换模型请加 --force 全量重建。"
        )

    if force:
        for paper in list(store.iter_papers()):
            store.delete_paper(paper.id)

    pdfs = sorted(pdf_dir.rglob("*.pdf"))
    if not pdfs:
        progress("未找到 PDF 文件")
    seen: set[str] = set()

    for i, path in enumerate(pdfs, 1):
        rel = str(path)
        seen.add(rel)
        progress(f"[{i}/{len(pdfs)}] {path.name}")
        try:
            digest = _sha256(path)
        except OSError as exc:
            result["failed"] += 1
            progress(f"  读取失败：{exc}")
            continue

        existing = store.paper_by_path(rel)
        if existing is not None and existing.sha256 == digest:
            if not existing.has_text or store.get_chunks_by_paper(existing.id):
                result["unchanged"] += 1
                continue

        try:
            pages, meta = extract_pdf(path)
        except Exception as exc:
            result["failed"] += 1
            progress(f"  解析失败：{exc}")
            continue

        title, authors, year = guess_metadata(path, meta, pages)
        has_text = any(p.strip() for p in pages)

        chunks: list[Chunk] = []
        if has_text:
            pieces = chunk_text(pages)
            if pieces:
                vectors = embedder.embed([t for _, t in pieces])
                chunks = [
                    Chunk(None, 0, i, page, text, vectors[i])
                    for i, (page, text) in enumerate(pieces)
                ]
        else:
            result["skipped_no_text"] += 1

        if refine and llm is not None and llm.is_configured:
            try:
                refined = refine_metadata(llm, path.name, pages[0][:2000] if pages else "")
                if refined:
                    if refined.get("title"):
                        title = refined["title"]
                    if refined.get("authors"):
                        authors = refined["authors"]
                    if refined.get("year") is not None:
                        year = refined["year"]
            except Exception:
                pass  # 提炼失败静默保留原值

        paper = Paper(
            id=None,
            path=rel,
            sha256=digest,
            title=title,
            authors=authors,
            year=year,
            page_count=len(pages),
            has_text=has_text,
            indexed_at=now_iso(),
        )
        store.upsert_paper(paper, chunks)
        result["added" if existing is None else "updated"] += 1

    if prune:
        known = {str(p) for p in pdfs}
        for paper in list(store.iter_papers()):
            if paper.path not in known:
                store.delete_paper(paper.id)
                result["removed"] += 1

    store.meta_set("embed_model", embedder.model_name)
    store.meta_set("library_dir", str(pdf_dir))
    return result
