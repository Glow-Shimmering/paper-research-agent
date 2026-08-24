"""Multi-provider aggregation and canonical source persistence."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Iterable, Mapping, Optional

from pragent.models import ResearchSource

from .base import MergedSource, NormalizedSource, SourceProvider, SourceProviderError
from .identity import deduplicate_sources


@dataclass(frozen=True)
class ProviderFailure:
    provider: str
    message: str
    code: str
    retryable: bool


@dataclass(frozen=True)
class DiscoveryItem:
    merged: MergedSource
    persisted: Optional[ResearchSource] = None


@dataclass(frozen=True)
class DiscoveryBatch:
    items: tuple[DiscoveryItem, ...]
    failures: tuple[ProviderFailure, ...]
    provider_counts: Mapping[str, int]


class DiscoveryService:
    def __init__(self, providers: Iterable[SourceProvider], *, repository=None) -> None:
        by_name: dict[str, SourceProvider] = {}
        for provider in providers:
            name = str(provider.name).strip().lower()
            if not name:
                raise ValueError("provider name 不能为空")
            if name in by_name:
                raise ValueError(f"重复 provider：{name}")
            by_name[name] = provider
        if not by_name:
            raise ValueError("至少需要一个 source provider")
        self.providers = by_name
        self.repository = repository

    def search(
        self,
        query: str,
        *,
        provider_names: Optional[Iterable[str]] = None,
        limit_per_provider: int = 10,
        persist: bool = True,
    ) -> DiscoveryBatch:
        query = str(query).strip()
        if not query:
            raise ValueError("query 不能为空")
        if (
            isinstance(limit_per_provider, bool)
            or not isinstance(limit_per_provider, int)
            or not 1 <= limit_per_provider <= 100
        ):
            raise ValueError("limit_per_provider 必须是 1–100 的整数")
        selected_names = (
            sorted(self.providers)
            if provider_names is None
            else sorted({str(name).strip().lower() for name in provider_names})
        )
        if not selected_names:
            raise ValueError("至少选择一个 provider")
        unknown = [name for name in selected_names if name not in self.providers]
        if unknown:
            raise ValueError(f"未知 provider：{', '.join(unknown)}")

        records: list[NormalizedSource] = []
        failures: list[ProviderFailure] = []
        counts: dict[str, int] = {name: 0 for name in selected_names}
        max_workers = min(4, len(selected_names))
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="pra-discovery") as pool:
            futures = {
                pool.submit(
                    self.providers[name].search,
                    query,
                    limit=limit_per_provider,
                ): name
                for name in selected_names
            }
            for future in as_completed(futures):
                name = futures[future]
                try:
                    provider_records = future.result()
                except SourceProviderError as exc:
                    failures.append(
                        ProviderFailure(
                            provider=name,
                            message=str(exc),
                            code=exc.code,
                            retryable=exc.retryable,
                        )
                    )
                except Exception as exc:
                    failures.append(
                        ProviderFailure(
                            provider=name,
                            message=f"{name} provider 失败：{exc}",
                            code="provider_error",
                            retryable=False,
                        )
                    )
                else:
                    counts[name] = len(provider_records)
                    records.extend(provider_records)

        merged_items = deduplicate_sources(records)
        items: list[DiscoveryItem] = []
        for merged in merged_items:
            persisted = None
            if persist:
                if self.repository is None:
                    raise RuntimeError("persist=True 需要 ResearchRepository")
                persisted = self.repository.upsert_merged_source(merged)
            items.append(DiscoveryItem(merged=merged, persisted=persisted))
        return DiscoveryBatch(
            items=tuple(items),
            failures=tuple(sorted(failures, key=lambda item: item.provider)),
            provider_counts={name: counts[name] for name in sorted(counts)},
        )
