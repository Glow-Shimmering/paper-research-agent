import pytest

from pragent.sources import (
    DiscoveryService,
    NormalizedSource,
    SourceProviderError,
    deduplicate_sources,
)
from pragent.storage import ResearchRepository
from pragent.store import Store


class FakeProvider:
    def __init__(self, name, records=None, error=None):
        self.name = name
        self.records = list(records or [])
        self.error = error
        self.calls = []

    def search(self, query, *, limit=10):
        self.calls.append((query, limit))
        if self.error:
            raise self.error
        return list(self.records)

    def lookup(self, identifier):
        return None


def source(provider, record_id, **kwargs):
    return NormalizedSource(
        provider=provider,
        provider_record_id=record_id,
        metadata={"provider": provider, "record_id": record_id},
        **kwargs,
    )


def test_multi_provider_discovery_dedupes_and_persists_all_provenance(tmp_path):
    db_path = tmp_path / "discovery.db"
    Store(db_path).close()
    repository = ResearchRepository(db_path)
    semantic = FakeProvider(
        "semantic_scholar",
        [
            source(
                "semantic_scholar",
                "Corpus-1",
                title="Semantic title",
                doi="10.1000/RAG",
                canonical_url="https://semanticscholar.org/paper/1",
            )
        ],
    )
    crossref = FakeProvider(
        "crossref",
        [
            source(
                "crossref",
                "10.1000/rag",
                title="Canonical Crossref title",
                authors=("Alice",),
                year=2024,
                doi="https://doi.org/10.1000/rag",
            )
        ],
    )
    failed = FakeProvider(
        "failing",
        error=SourceProviderError(
            "fixture unavailable",
            provider="failing",
            code="http_503",
            retryable=True,
            status_code=503,
        ),
    )

    batch = DiscoveryService(
        [semantic, crossref, failed], repository=repository
    ).search("retrieval agents", limit_per_provider=5)

    assert len(batch.items) == 1
    item = batch.items[0]
    assert item.merged.canonical_key == "doi:10.1000/rag"
    assert item.merged.duplicate_count == 1
    assert item.persisted.title == "Canonical Crossref title"
    assert repository.list_sources().total == 1
    assert {
        record.provider for record in repository.list_source_records(item.persisted.id)
    } == {"semantic_scholar", "crossref"}
    assert repository.list_source_identities(item.persisted.id)[0].normalized_value == (
        "10.1000/rag"
    )
    assert batch.provider_counts == {
        "crossref": 1,
        "failing": 0,
        "semantic_scholar": 1,
    }
    assert batch.failures[0].retryable is True
    repository.close()


def test_bridge_identity_merges_existing_sources_and_project_memberships(tmp_path):
    db_path = tmp_path / "bridge.db"
    Store(db_path).close()
    repository = ResearchRepository(db_path)
    project = repository.create_project("去重项目")

    url_only = deduplicate_sources(
        [
            source(
                "semantic_scholar",
                "S-1",
                title="URL source",
                canonical_url="https://example.org/work",
            )
        ]
    )[0]
    doi_only = deduplicate_sources(
        [source("crossref", "10.1000/bridge", title="DOI source", doi="10.1000/bridge")]
    )[0]
    first = repository.upsert_merged_source(url_only)
    second = repository.upsert_merged_source(doi_only)
    repository.add_project_source(project.id, first.id, position=4)
    repository.add_project_source(project.id, second.id, position=2)
    assert repository.list_sources().total == 2

    bridge = deduplicate_sources(
        [
            source(
                "bridge_provider",
                "bridge-1",
                title="Bridge",
                doi="10.1000/bridge",
                canonical_url="https://example.org/work/",
            )
        ]
    )[0]
    merged = repository.upsert_merged_source(bridge)

    assert repository.list_sources().total == 1
    assert merged.canonical_key == "doi:10.1000/bridge"
    memberships = repository.list_project_sources(project.id)
    assert memberships.total == 1
    assert memberships.items[0].source.id == merged.id
    assert memberships.items[0].position == 2
    assert {
        item.provider for item in repository.list_source_records(merged.id)
    } == {"semantic_scholar", "crossref", "bridge_provider"}
    assert {
        (item.identity_kind, item.normalized_value)
        for item in repository.list_source_identities(merged.id)
    } == {
        ("doi", "10.1000/bridge"),
        ("url", "https://example.org/work"),
    }
    repository.close()


def test_discovery_validates_provider_selection_and_persistence_boundary(tmp_path):
    provider = FakeProvider("one", [source("one", "1", title="No identity")])
    service = DiscoveryService([provider])

    with pytest.raises(ValueError, match="未知 provider"):
        service.search("query", provider_names=["missing"], persist=False)
    with pytest.raises(RuntimeError, match="ResearchRepository"):
        service.search("query")
    offline = service.search("query", persist=False)
    assert len(offline.items) == 1 and offline.items[0].persisted is None
