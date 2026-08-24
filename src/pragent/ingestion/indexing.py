"""Source-aware adapters into the existing paper/chunk/embed/search pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from pragent.chunking import chunk_text
from pragent.indexer import _check_embed_model, index_pdf
from pragent.models import Chunk, Paper, ResearchSource
from pragent.store import Store, now_iso


@dataclass(frozen=True)
class IndexedSourceResult:
    source: ResearchSource
    paper: Paper
    index_result: dict[str, int]
    deduplicated: bool = False


def index_web_source(
    store: Store,
    repository,
    source_id: str,
    embedder,
    *,
    progress: Callable[[str], None] = print,
) -> IndexedSourceResult:
    source = _require_indexable_source(repository, source_id, expected_kind="web")
    if not source.extracted_text or not source.snapshot_sha256:
        raise ValueError("网页来源缺少 extracted_text 或 snapshot_sha256")
    state = store.index_state()
    _check_embed_model(state, embedder, force=False)
    logical_path = f"pragent-web://{source.id}"
    existing = next((paper for paper in state.papers if paper.path == logical_path), None)
    locator = {
        "kind": "web_snapshot",
        "source_id": source.id,
        "snapshot_sha256": source.snapshot_sha256,
    }
    result = _empty_index_result()
    if (
        existing is not None
        and existing.sha256 == source.snapshot_sha256
        and existing.id in state.paper_ids_with_chunks
    ):
        result["unchanged"] = 1
        paper = store.set_paper_document_metadata(
            existing.id,
            source_kind="web",
            canonical_uri=source.canonical_url,
            locator=locator,
        )
    else:
        progress(f"索引网页来源：{source.title or source.canonical_url or source.id}")
        pieces = chunk_text([source.extracted_text])
        if not pieces:
            raise ValueError("网页抽取正文无法生成检索分块")
        vectors = embedder.embed([text for _, text in pieces])
        chunks = [
            Chunk(None, 0, sequence, page, text, vectors[sequence])
            for sequence, (page, text) in enumerate(pieces)
        ]
        paper_value = Paper(
            id=None,
            path=logical_path,
            sha256=source.snapshot_sha256,
            title=source.title or source.canonical_url or "网页来源",
            authors=list(source.authors),
            year=source.year,
            page_count=1,
            has_text=True,
            indexed_at=now_iso(),
            source_kind="web",
            canonical_uri=source.canonical_url,
            locator=locator,
        )
        store.commit_index_update(
            [(paper_value, chunks)],
            [],
            embed_model=embedder.model_name,
            library_dir=None,
            expected_revision=state.revision,
        )
        paper = store.paper_by_path(logical_path)
        result["added" if existing is None else "updated"] = 1
    if paper is None:  # pragma: no cover - successful commit invariant
        raise RuntimeError("网页文档索引后无法读取")
    linked = repository.attach_indexed_paper(
        source.id,
        paper.id,
        expected_version=source.version,
    )
    deduplicated = linked.indexed_paper_id != paper.id
    if deduplicated:
        store.delete_paper(paper.id)
        paper = store.paper_by_id(linked.indexed_paper_id)
        if paper is None:  # pragma: no cover - repository FK invariant
            raise RuntimeError("去重后的索引文档不存在")
    paper = store.set_paper_document_metadata(
        paper.id,
        source_kind="web",
        canonical_uri=linked.canonical_url,
        locator={
            "kind": "web_snapshot",
            "source_id": linked.id,
            "snapshot_sha256": linked.snapshot_sha256 or linked.content_sha256,
        },
    )
    return IndexedSourceResult(linked, paper, result, deduplicated)


def index_pdf_source(
    store: Store,
    repository,
    source_id: str,
    pdf_path: Path,
    embedder,
    *,
    refine: bool = False,
    llm=None,
    progress: Callable[[str], None] = print,
) -> IndexedSourceResult:
    source = _require_indexable_source(repository, source_id, expected_kind="paper")
    result = index_pdf(
        store,
        pdf_path,
        embedder,
        refine=refine,
        llm=llm,
        progress=progress,
    )
    if result["failed"]:
        raise RuntimeError("PDF 索引失败")
    resolved = pdf_path.expanduser().resolve(strict=True)
    paper = store.paper_by_path(str(resolved))
    if paper is None:
        raise RuntimeError("PDF 索引后无法读取")
    paper = store.set_paper_document_metadata(
        paper.id,
        source_kind="pdf",
        canonical_uri=source.canonical_url,
        locator={"kind": "pdf", "source_id": source.id},
    )
    linked = repository.attach_indexed_paper(
        source.id,
        paper.id,
        expected_version=source.version,
    )
    deduplicated = linked.indexed_paper_id != paper.id
    if deduplicated:
        store.delete_paper(paper.id)
        paper = store.paper_by_id(linked.indexed_paper_id)
        if paper is None:  # pragma: no cover - repository FK invariant
            raise RuntimeError("去重后的 PDF 索引文档不存在")
    paper = store.set_paper_document_metadata(
        paper.id,
        source_kind="pdf",
        canonical_uri=linked.canonical_url,
        locator={"kind": "pdf", "source_id": linked.id},
    )
    return IndexedSourceResult(linked, paper, result, deduplicated)


def _require_indexable_source(repository, source_id: str, *, expected_kind: str):
    source = repository.get_source(source_id)
    if source is None:
        raise KeyError(f"研究来源不存在：{source_id}")
    if source.source_kind != expected_kind:
        raise ValueError(f"来源 {source_id} 不是 {expected_kind} 类型")
    if source.status not in {"ready", "discovered", "fetching", "failed"}:
        raise ValueError(f"来源当前状态不能索引：{source.status}")
    return source


def _empty_index_result() -> dict[str, int]:
    return {
        "added": 0,
        "updated": 0,
        "unchanged": 0,
        "failed": 0,
        "removed": 0,
        "skipped_no_text": 0,
    }
