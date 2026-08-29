"""索引管线：扫描 PDF 目录 → 解析 → 分块 → 嵌入 → 入库。"""
import hashlib
from pathlib import Path
from typing import Callable, Optional

from .chunking import chunk_text
from .llm import refine_metadata
from .models import Chunk, Paper
from .pdf import extract_pdf, guess_metadata
from .store import IndexState, Store, now_iso


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def validate_pdf_directory(pdf_dir: Path) -> Path:
    """返回绝对、规范化的论文目录；无效路径在接触索引库前失败。"""
    candidate = pdf_dir.expanduser()
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise RuntimeError(f"论文目录不存在或无法访问：{candidate}") from exc
    if not resolved.is_dir():
        raise RuntimeError(f"论文目录不是文件夹：{resolved}")
    return resolved


def _validate_pdf_file(pdf_path: Path) -> Path:
    candidate = pdf_path.expanduser()
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise RuntimeError(f"PDF 文件不存在或无法访问：{candidate}") from exc
    if not resolved.is_file():
        raise RuntimeError(f"PDF 路径不是文件：{resolved}")
    if resolved.suffix.lower() != ".pdf":
        raise RuntimeError(f"文件不是 PDF：{resolved}")
    return resolved


def _normalized_path(path: str | Path) -> Path:
    """规范化数据库中的旧路径；目标已删除时也可比较其应在的目录。"""
    return Path(path).expanduser().resolve(strict=False)


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
        return True
    except ValueError:
        return False


def _inherit_document_metadata(paper: Paper, existing: Optional[Paper]) -> Paper:
    if existing is not None:
        paper.source_kind = existing.source_kind
        paper.canonical_uri = existing.canonical_uri
        paper.locator = existing.locator
    return paper


def _force_scope_ids(
    papers: list[Paper],
    *,
    previous_library_dir: Optional[str],
    current_library_dir: Path,
    seen_paths: set[str],
) -> list[int]:
    roots = [current_library_dir]
    if previous_library_dir:
        previous = _normalized_path(previous_library_dir)
        if previous not in roots:
            roots.append(previous)
    delete_ids: list[int] = []
    for paper in papers:
        if paper.source_kind == "web" or paper.path in seen_paths:
            continue
        path = _normalized_path(paper.path)
        in_scope = (
            any(_is_within(path, root) for root in roots)
            if previous_library_dir
            else paper.source_kind == "pdf"
        )
        if in_scope and paper.id is not None:
            delete_ids.append(paper.id)
    return delete_ids


def _new_result() -> dict:
    return {
        "added": 0,
        "updated": 0,
        "unchanged": 0,
        "failed": 0,
        "removed": 0,
        "skipped_no_text": 0,
    }


def _prepare_paper(
    path: Path,
    digest: str,
    embedder,
    *,
    refine: bool,
    llm,
) -> tuple[Paper, list[Chunk], bool]:
    """解析并嵌入一篇论文，但不写数据库。"""
    pages, meta = extract_pdf(path)
    title, authors, year = guess_metadata(path, meta, pages)
    has_text = any(p.strip() for p in pages)

    chunks: list[Chunk] = []
    if has_text:
        pieces = chunk_text(pages)
        if pieces:
            vectors = embedder.embed([text for _, text in pieces])
            chunks = [
                Chunk(None, 0, seq, page, text, vectors[seq])
                for seq, (page, text) in enumerate(pieces)
            ]

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

    return (
        Paper(
            id=None,
            path=str(path),
            sha256=digest,
            title=title,
            authors=authors,
            year=year,
            page_count=len(pages),
            has_text=has_text,
            indexed_at=now_iso(),
        ),
        chunks,
        not has_text,
    )


def _check_embed_model(state: IndexState, embedder, *, force: bool) -> None:
    stored_model = state.embed_model
    if state.has_search_corpus and not stored_model and not force:
        raise RuntimeError(
            "索引中已有向量但缺少嵌入模型来源，无法安全增量更新。"
            "请使用 --force 全量重建。"
        )
    if stored_model is not None and stored_model != embedder.model_name and not force:
        raise RuntimeError(
            f"库由嵌入模型「{stored_model}」建立，当前为「{embedder.model_name}」。"
            "如需切换模型请加 --force 全量重建。"
        )


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
    should_continue: Optional[Callable[[], bool]] = None,
) -> dict:
    """增量索引。返回 {added, updated, unchanged, failed, removed, skipped_no_text}。

    ``should_continue`` 为可选的逐篇继续检查（取消/预算）；返回 False 时
    停止处理剩余文件并把 ``cancelled`` 置入结果。已处理部分仍以一致的
    事务提交，重跑可安全续作。
    """
    result = _new_result()
    pdf_dir = validate_pdf_directory(pdf_dir)
    state = store.index_state()
    existing_papers = list(state.papers)
    existing_by_path = {paper.path: paper for paper in existing_papers}
    _check_embed_model(state, embedder, force=force)

    stored_library_dir = state.library_dir
    if existing_papers and prune and not force and not stored_library_dir:
        raise RuntimeError(
            "现有索引缺少论文目录记录，无法安全判断清理范围；"
            "请使用 --no-prune 增量修复，或使用 --force 明确全量替换。"
        )
    if stored_library_dir and existing_papers and prune and not force:
        previous_dir = _normalized_path(stored_library_dir)
        if previous_dir != pdf_dir:
            raise RuntimeError(
                f"当前目录「{pdf_dir}」与已索引论文目录「{previous_dir}」不同；"
                "为避免误删旧索引，请使用 --no-prune 增量添加，或使用 --force 明确全量替换。"
            )

    pdfs = sorted(path.resolve() for path in pdf_dir.rglob("*.pdf") if path.is_file())
    if not pdfs:
        progress("未找到 PDF 文件")
        if existing_papers and (prune or force):
            raise RuntimeError(
                f"目录「{pdf_dir}」中未找到 PDF；为避免清空现有索引，已取消操作。"
                "如需仅检查空目录，请使用 --no-prune。"
            )
    seen: set[str] = set()

    # --force 先完成所有读取、解析和嵌入。任一论文失败时旧库完全不动。
    if force:
        staged: list[tuple[Paper, list[Chunk], bool]] = []
        for i, path in enumerate(pdfs, 1):
            if should_continue is not None and not should_continue():
                # force 是全量替换：取消后中止，原索引完全不动。
                raise RuntimeError("强制重建在取消信号后中止；原索引未修改。")
            seen.add(str(path))
            progress(f"[{i}/{len(pdfs)}] {path.name}")
            try:
                digest = _sha256(path)
                paper, chunks, skipped = _prepare_paper(
                    path, digest, embedder, refine=refine, llm=llm
                )
                staged.append(
                    (
                        _inherit_document_metadata(
                            paper, existing_by_path.get(str(path))
                        ),
                        chunks,
                        skipped,
                    )
                )
            except Exception as exc:
                result["failed"] += 1
                progress(f"  处理失败：{exc}")
        if result["failed"]:
            raise RuntimeError(
                f"强制重建预处理失败（{result['failed']} 篇）；原索引未删除。"
            )

        # Force 只替换主 PDF scope；Web 和主目录外的显式来源继续保留，
        # 但必须在同一提交中用当前模型重嵌入，避免混合向量模型。
        scope_delete_ids = _force_scope_ids(
            existing_papers,
            previous_library_dir=stored_library_dir,
            current_library_dir=pdf_dir,
            seen_paths=seen,
        )
        delete_set = set(scope_delete_ids)
        preserved_entries: list[tuple[Paper, list[Chunk]]] = []
        try:
            for existing in existing_papers:
                if existing.id in delete_set or existing.path in seen:
                    continue
                existing_chunks = store.paper_chunks(existing.id)
                if existing_chunks:
                    vectors = embedder.embed([chunk.text for chunk in existing_chunks])
                    existing_chunks = [
                        Chunk(
                            None,
                            existing.id,
                            chunk.seq,
                            chunk.page,
                            chunk.text,
                            vectors[index],
                        )
                        for index, chunk in enumerate(existing_chunks)
                    ]
                preserved_entries.append((existing, existing_chunks))
        except Exception as exc:
            raise RuntimeError(
                "强制重建无法重嵌入 scope 外文档；原索引未删除。"
            ) from exc
        store.commit_index_update(
            [(paper, chunks) for paper, chunks, _ in staged]
            + preserved_entries,
            scope_delete_ids,
            embed_model=embedder.model_name,
            library_dir=str(pdf_dir),
            expected_revision=state.revision,
        )
        for paper, chunks, skipped_no_text in staged:
            result["added"] += 1
            if skipped_no_text:
                result["skipped_no_text"] += 1
        return result

    staged_updates: list[tuple[Paper, list[Chunk]]] = []
    for i, path in enumerate(pdfs, 1):
        if should_continue is not None and not should_continue():
            result["cancelled"] = True
            progress("收到取消信号；停止处理剩余文件，已处理部分保持一致")
            break
        normalized = str(path)
        seen.add(normalized)
        progress(f"[{i}/{len(pdfs)}] {path.name}")
        try:
            digest = _sha256(path)
        except OSError as exc:
            result["failed"] += 1
            progress(f"  读取失败：{exc}")
            continue

        existing = existing_by_path.get(normalized)
        if existing is not None and existing.sha256 == digest:
            if not existing.has_text or existing.id in state.paper_ids_with_chunks:
                result["unchanged"] += 1
                continue

        try:
            paper, chunks, skipped_no_text = _prepare_paper(
                path, digest, embedder, refine=refine, llm=llm
            )
            paper = _inherit_document_metadata(paper, existing)
        except Exception as exc:
            result["failed"] += 1
            progress(f"  处理失败：{exc}")
            continue
        if skipped_no_text:
            result["skipped_no_text"] += 1
        staged_updates.append((paper, chunks))
        result["added" if existing is None else "updated"] += 1

    delete_paper_ids: list[int] = []
    if prune:
        known = set(seen)
        for paper in existing_papers:
            paper_path = _normalized_path(paper.path)
            # 单文件下载等增量来源可能不在主论文目录中，不应被主目录清理误删。
            if _is_within(paper_path, pdf_dir) and paper.path not in known:
                delete_paper_ids.append(paper.id)
                result["removed"] += 1

    library_dir_update: Optional[str] = None
    if not stored_library_dir or _normalized_path(stored_library_dir) == pdf_dir or prune:
        library_dir_update = str(pdf_dir)
    store.commit_index_update(
        staged_updates,
        delete_paper_ids,
        embed_model=embedder.model_name,
        library_dir=library_dir_update,
        expected_revision=state.revision,
    )
    return result


def index_pdf(
    store: Store,
    pdf_path: Path,
    embedder,
    *,
    refine: bool = False,
    llm=None,
    set_library_dir_if_missing: bool = False,
    progress: Callable[[str], None] = print,
) -> dict:
    """只增量索引一个 PDF；不扫描目录、不清理其他论文。"""
    result = _new_result()
    path = _validate_pdf_file(pdf_path)
    state = store.index_state()
    _check_embed_model(state, embedder, force=False)
    progress(f"[1/1] {path.name}")

    try:
        digest = _sha256(path)
    except OSError as exc:
        result["failed"] = 1
        progress(f"  读取失败：{exc}")
        return result

    existing = next((paper for paper in state.papers if paper.path == str(path)), None)
    if existing is not None and existing.sha256 == digest:
        if not existing.has_text or existing.id in state.paper_ids_with_chunks:
            result["unchanged"] = 1

    staged_updates: list[tuple[Paper, list[Chunk]]] = []
    if not result["unchanged"]:
        try:
            paper, chunks, skipped_no_text = _prepare_paper(
                path, digest, embedder, refine=refine, llm=llm
            )
            paper = _inherit_document_metadata(paper, existing)
        except Exception as exc:
            result["failed"] = 1
            progress(f"  处理失败：{exc}")
            return result
        if skipped_no_text:
            result["skipped_no_text"] = 1
        staged_updates.append((paper, chunks))
        result["added" if existing is None else "updated"] = 1

    library_dir_update = (
        str(path.parent) if set_library_dir_if_missing and not state.library_dir else None
    )
    store.commit_index_update(
        staged_updates,
        [],
        embed_model=embedder.model_name,
        library_dir=library_dir_update,
        expected_revision=state.revision,
    )
    return result
