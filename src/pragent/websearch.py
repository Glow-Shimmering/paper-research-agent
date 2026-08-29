"""兼容联网论文搜索 API；实现已迁移到统一 arXiv adapter。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .sources.arxiv import ArxivAdapter
from .sources.base import SourceProviderError


class WebSearchError(Exception):
    pass


@dataclass
class WebPaper:
    title: str
    authors: list[str]
    year: Optional[int]
    abstract: str
    url: str
    pdf_url: Optional[str]


_ADAPTER = ArxivAdapter()


def search_papers(query: str, limit: int = 5, timeout: Optional[float] = None) -> list[WebPaper]:
    """按相关性检索 arXiv；保留旧 ``WebPaper`` 返回合同。

    ``timeout`` 为本次调用的网络预算（秒）；``None`` 时使用 adapter 默认值。
    """

    try:
        if timeout is None:
            records = _ADAPTER.search(query, limit=limit)
        else:
            records = _ADAPTER.search(query, limit=limit, timeout=timeout)
    except SourceProviderError as exc:
        raise WebSearchError(str(exc)) from exc
    return [
        WebPaper(
            title=record.title,
            authors=list(record.authors),
            year=record.year,
            abstract=record.abstract,
            url=record.landing_url or record.canonical_url or "",
            pdf_url=record.pdf_url,
        )
        for record in records
    ]
