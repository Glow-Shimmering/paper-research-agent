"""Explicit user-triggered fetch/download/index actions for persisted sources."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Optional

from pragent.download import download_pdf
from pragent.ingestion.indexing import IndexedSourceResult, index_pdf_source, index_web_source
from pragent.models import ProjectSourceMembership, ResearchSource
from pragent.storage import RecordVersionConflictError


class SourceActionError(RuntimeError):
    def __init__(self, message: str, *, code: str, retryable: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable


@dataclass(frozen=True)
class SourceActionResult:
    source: ResearchSource
    indexed: IndexedSourceResult
    membership: Optional[ProjectSourceMembership] = None


class SourceActionService:
    def __init__(
        self,
        repository,
        store,
        embedder,
        *,
        web_ingest,
        download_directory: str | Path,
        downloader: Callable = download_pdf,
    ) -> None:
        self.repository = repository
        self.store = store
        self.embedder = embedder
        self.web_ingest = web_ingest
        self.download_directory = Path(download_directory).expanduser().resolve(
            strict=False
        )
        self.downloader = downloader

    def import_web(
        self, url: str, *, project_id: Optional[str] = None
    ) -> SourceActionResult:
        self._validate_project(project_id)
        url = str(url).strip()
        if not url or len(url) > 2_000:
            raise SourceActionError(
                "网页 URL 不能为空且不能超过 2000 字符", code="invalid_url"
            )
        try:
            ingested = self.web_ingest.ingest(url)
            indexed = index_web_source(
                self.store,
                self.repository,
                ingested.source.id,
                self.embedder,
                progress=lambda _message: None,
            )
        except Exception as exc:
            code = str(getattr(exc, "code", "web_ingest_failed"))
            raise SourceActionError(
                f"网页抓取或索引失败（{code}）",
                code=code,
                retryable=bool(getattr(exc, "retryable", True)),
            ) from exc
        membership = self._add_to_project(project_id, indexed.source.id)
        return SourceActionResult(indexed.source, indexed, membership)

    def download_and_index(
        self, source_id: str, *, project_id: Optional[str] = None
    ) -> SourceActionResult:
        self._validate_project(project_id)
        source = self.repository.get_source(source_id)
        if source is None:
            raise SourceActionError("研究来源不存在", code="source_not_found")
        if source.source_kind != "paper":
            raise SourceActionError("该来源不是论文", code="not_a_paper")
        if not source.arxiv_id:
            raise SourceActionError(
                "当前仅支持下载具有 arXiv ID 的 PDF",
                code="download_unavailable",
            )
        try:
            source = self.repository.update_source(
                source.id,
                expected_version=source.version,
                status="fetching",
            )
        except RecordVersionConflictError as exc:
            raise SourceActionError(
                "来源状态已变化，请刷新后重试", code="source_busy"
            ) from exc
        try:
            self.download_directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            pdf_path = self.downloader(
                f"https://arxiv.org/abs/{source.arxiv_id}",
                self.download_directory,
            )
            indexed = index_pdf_source(
                self.store,
                self.repository,
                source.id,
                Path(pdf_path),
                self.embedder,
                progress=lambda _message: None,
            )
            cleared = self._clear_error(indexed.source)
            if cleared is not indexed.source:
                indexed = replace(indexed, source=cleared)
        except Exception as exc:
            self._mark_failed(source.id, "pdf_download_or_index_failed")
            raise SourceActionError(
                "PDF 下载或索引失败，请检查来源与本地下载目录",
                code="pdf_download_or_index_failed",
                retryable=True,
            ) from exc
        membership = self._add_to_project(project_id, indexed.source.id)
        return SourceActionResult(indexed.source, indexed, membership)

    def _validate_project(self, project_id: Optional[str]) -> None:
        if project_id and self.repository.get_project(project_id) is None:
            raise SourceActionError(
                "研究项目不存在", code="project_not_found"
            )

    def _add_to_project(
        self, project_id: Optional[str], source_id: str
    ) -> Optional[ProjectSourceMembership]:
        if not project_id:
            return None
        try:
            return self.repository.add_project_source(project_id, source_id)
        except KeyError as exc:
            raise SourceActionError(
                "研究项目不存在", code="project_not_found"
            ) from exc

    def _clear_error(self, source: ResearchSource) -> ResearchSource:
        if not isinstance(source.metadata, dict) or "last_error" not in source.metadata:
            return source
        metadata = dict(source.metadata)
        metadata.pop("last_error", None)
        return self.repository.update_source(
            source.id,
            expected_version=source.version,
            metadata=metadata,
        )

    def _mark_failed(self, source_id: str, code: str) -> None:
        current = self.repository.get_source(source_id)
        if current is None:
            return
        metadata = dict(current.metadata)
        metadata["last_error"] = {
            "code": code,
            "message": "操作失败，可在 Library 中重试",
        }
        try:
            self.repository.update_source(
                source_id,
                expected_version=current.version,
                status="failed",
                metadata=metadata,
            )
        except Exception:
            return
