import re

from fastapi.testclient import TestClient as FastAPITestClient

from pragent.ingestion.safe_fetch import SafeFetchError, SafeFetchResult
from pragent.ingestion.snapshots import SnapshotStore
from pragent.sources import NormalizedSource, SourceProviderError
from pragent.storage import ResearchRepository
from pragent.store import Store
from pragent.webapp import create_app

from helpers import FakeEmbedder, make_pdf

ARTICLE = b'''<!doctype html><html><head>
<meta name="citation_title" content="Public Evidence Report">
<meta name="citation_author" content="Web Author">
<meta name="citation_date" content="2025-01-01">
</head><body><article>
<h1>Public Evidence Report</h1>
<p>provider discovery evidence is available in this public web report.</p>
<p>The report compares canonical identity merging and retrieval quality.</p>
<p>Limitations remain visible and every source should retain provenance.</p>
</article></body></html>'''


class OfflineLLM:
    is_configured = False


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


class MutableFetcher:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = []

    def fetch(self, url):
        self.calls.append(url)
        if self.error:
            raise self.error
        return self.result


def TestClient(app, **kwargs):
    kwargs.setdefault("base_url", "http://127.0.0.1")
    return FastAPITestClient(app, **kwargs)


def provider_records():
    semantic = NormalizedSource(
        provider="semantic_scholar",
        provider_record_id="Corpus-42",
        title="Semantic Provider Title",
        authors=("Alice",),
        year=2025,
        abstract="provider discovery evidence",
        doi="10.1000/discovery",
        arxiv_id="2501.00001v2",
        canonical_url="https://semanticscholar.org/paper/42",
        pdf_url="https://arxiv.org/pdf/2501.00001",
        metadata={"paperId": "Corpus-42", "openAccessPdf": {"url": "https://arxiv.org/pdf/2501.00001"}},
    )
    crossref = NormalizedSource(
        provider="crossref",
        provider_record_id="10.1000/discovery",
        title="Canonical Discovery Paper",
        authors=("Alice", "Bob"),
        year=2025,
        doi="https://doi.org/10.1000/DISCOVERY",
        canonical_url="https://doi.org/10.1000/discovery",
        metadata={"DOI": "10.1000/discovery"},
    )
    return semantic, crossref


def fetched(url="https://example.org/report"):
    return SafeFetchResult(
        requested_url=url,
        final_url=url,
        status_code=200,
        content_type="text/html",
        body=ARTICLE,
        redirect_chain=(),
        resolved_ips=("93.184.216.34",),
    )


def make_app(tmp_path, *, fetcher=None, failing_provider=True):
    store = Store(tmp_path / "discovery-web.db")
    repository = ResearchRepository(tmp_path / "discovery-web.db")
    semantic, crossref = provider_records()
    providers = [
        FakeProvider("semantic_scholar", [semantic]),
        FakeProvider("crossref", [crossref]),
    ]
    if failing_provider:
        providers.append(
            FakeProvider(
                "fixture_failure",
                error=SourceProviderError(
                    "fixture provider unavailable",
                    provider="fixture_failure",
                    code="http_503",
                    retryable=True,
                    status_code=503,
                ),
            )
        )
    fetcher = fetcher or MutableFetcher(fetched())
    download_calls = []

    def downloader(url, target_dir):
        download_calls.append((url, target_dir))
        return make_pdf(
            target_dir / "2501.00001.pdf",
            ["provider discovery evidence appears in the downloaded PDF. " * 20],
            {"title": "Canonical Discovery Paper", "author": "Alice"},
        )

    app = create_app(
        store=store,
        research_repository=repository,
        embedder=FakeEmbedder(),
        llm=OfflineLLM(),
        source_providers=providers,
        web_fetcher=fetcher,
        snapshot_store=SnapshotStore(tmp_path / "snapshots"),
        pdf_downloader=downloader,
        download_directory=tmp_path / "downloads",
    )
    return app, store, repository, providers, fetcher, download_calls


def test_discovery_json_dedupes_persists_downloads_fetches_and_redacts(tmp_path):
    app, store, repository, providers, fetcher, download_calls = make_app(tmp_path)
    with TestClient(app) as client:
        project = client.post("/api/v1/projects", json={"title": "发现项目"}).json()
        response = client.post(
            "/api/v1/discover/search",
            json={
                "query": "provider discovery",
                "providers": ["semantic_scholar", "crossref", "fixture_failure"],
                "limit": 5,
            },
        )
        assert response.status_code == 200
        discovery = response.json()
        assert len(discovery["items"]) == 1
        item = discovery["items"][0]
        source = item["source"]
        assert source["title"] == "Canonical Discovery Paper"
        assert source["doi"] == "10.1000/discovery"
        assert source["arxiv_id"] == "2501.00001"
        assert item["duplicate_count"] == 1
        assert set(item["providers"]) == {"crossref", "semantic_scholar"}
        assert discovery["failures"][0]["code"] == "http_503"
        assert "metadata" not in source and "snapshot_path" not in source

        selected = client.post(
            f"/api/v1/projects/{project['id']}/sources/{source['id']}"
        )
        assert selected.status_code == 201
        downloaded = client.post(
            f"/api/v1/sources/{source['id']}/download",
            json={"project_id": project["id"]},
        )
        assert downloaded.status_code == 200
        assert downloaded.json()["source"]["indexed"] is True
        assert download_calls and download_calls[0][0].endswith("2501.00001")

        imported = client.post(
            "/api/v1/sources/web",
            json={"url": "https://example.org/report", "project_id": project["id"]},
        )
        assert imported.status_code == 201
        assert imported.json()["source"]["source_kind"] == "web"
        assert imported.json()["source"]["indexed"] is True
        assert fetcher.calls == ["https://example.org/report"]

        hits = client.get(
            "/api/search", params={"q": "provider discovery evidence", "top": 20}
        ).json()["hits"]
        assert {hit["source_kind"] for hit in hits} == {"pdf", "web"}
        library = client.get("/api/v1/sources").json()
        project_sources = client.get(
            f"/api/v1/projects/{project['id']}/sources"
        ).json()

    serialized = str(
        {
            "discovery": discovery,
            "downloaded": downloaded.json(),
            "imported": imported.json(),
            "hits": hits,
            "library": library,
            "project_sources": project_sources,
        }
    )
    assert str(tmp_path) not in serialized
    for forbidden in ("snapshot_path", "extracted_text", "locator", "raw_metadata"):
        assert forbidden not in serialized
    assert library["total"] == 2
    assert project_sources["total"] == 2
    repository.close()
    store.close()


def test_discover_and_library_htmx_show_badges_partial_failure_and_selection(tmp_path):
    app, store, repository, _, _, _ = make_app(tmp_path)
    with TestClient(app, follow_redirects=False) as client:
        page = client.get("/ui/discover")
        assert page.status_code == 200
        assert "多来源论文发现" in page.text
        csrf = client.cookies["pra_csrf"]
        project = client.post(
            "/api/v1/projects", json={"title": "HTMX 发现项目"}
        ).json()

        results = client.post(
            "/ui/discover/search",
            data={
                "csrf_token": csrf,
                "query": "provider discovery",
                "limit": "5",
                "project_id": project["id"],
                "provider_semantic_scholar": "on",
                "provider_crossref": "on",
                "provider_fixture_failure": "on",
            },
            headers={"HX-Request": "true"},
        )
        assert results.status_code == 200
        assert "Canonical Discovery Paper" in results.text
        assert "合并 2 条记录" in results.text
        assert "fixture provider unavailable" in results.text
        assert "semantic_scholar" in results.text and "crossref" in results.text
        source_id = re.search(r"/sources/(source_[a-f0-9]+)/download", results.text).group(1)

        selected = client.post(
            f"/ui/projects/{project['id']}/sources/{source_id}",
            data={"csrf_token": csrf},
            headers={"HX-Request": "true"},
        )
        assert selected.status_code == 200
        assert "来源已加入研究项目" in selected.text

        library = client.get("/ui/library")
        assert library.status_code == 200
        assert "统一来源库" in library.text
        assert "Canonical Discovery Paper" in library.text
        assert str(tmp_path) not in library.text
        assert "snapshot_path" not in library.text
    repository.close()
    store.close()


def test_library_never_turns_unvalidated_internal_url_into_a_link(tmp_path):
    app, store, repository, _, _, _ = make_app(
        tmp_path, failing_provider=False
    )
    unsafe = repository.create_source(
        "record:unsafe",
        "web",
        title="Unsafe legacy metadata",
        canonical_url="javascript:alert(1)",
        status="failed",
    )
    repository.add_source_record(
        unsafe.id, "fixture", "unsafe", {"url": "javascript:alert(1)"}
    )
    with TestClient(app) as client:
        api_source = client.get("/api/v1/sources").json()["items"][0]
        page = client.get("/ui/library")

    assert api_source["canonical_url"] is None
    assert "javascript:alert" not in page.text
    repository.close()
    store.close()


def test_failed_web_import_is_visible_and_library_retry_recovers(tmp_path):
    timeout = SafeFetchError("fixture timeout", code="timeout", retryable=True)
    fetcher = MutableFetcher(error=timeout)
    app, store, repository, _, _, _ = make_app(
        tmp_path, fetcher=fetcher, failing_provider=False
    )
    with TestClient(app) as client:
        page = client.get("/ui/discover")
        csrf = client.cookies["pra_csrf"]
        failed = client.post(
            "/ui/discover/web",
            data={
                "csrf_token": csrf,
                "url": "https://example.org/retry",
                "project_id": "",
            },
            headers={"HX-Request": "true"},
        )
        assert failed.status_code == 502
        assert "timeout" in failed.text
        source = repository.list_sources().items[0]
        assert source.status == "failed"

        fetcher.error = None
        fetcher.result = fetched("https://example.org/retry")
        recovered = client.post(
            f"/ui/library/{source.id}/retry",
            data={"csrf_token": csrf, "project_id": ""},
            headers={"HX-Request": "true"},
        )
        assert recovered.status_code == 200
        assert "来源已恢复并完成索引" in recovered.text
        assert repository.get_source(source.id).status == "ready"
        library = client.get("/ui/library")
        assert "全文已索引" in library.text
        assert "timeout" not in library.text
    repository.close()
    store.close()
