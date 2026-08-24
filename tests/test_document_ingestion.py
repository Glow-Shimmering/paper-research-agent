from fastapi.testclient import TestClient

from pragent.indexer import index_library
from pragent.ingestion.indexing import index_pdf_source, index_web_source
from pragent.search import hybrid_search
from pragent.storage import ResearchRepository
from pragent.store import Store
from pragent.webapp import create_app

from helpers import FakeEmbedder, make_pdf, noop_progress


class OfflineLLM:
    is_configured = False


def create_web_source(
    repository,
    *,
    text,
    sha="a" * 64,
    url="https://example.org/report",
    add_content_identity=True,
):
    source = repository.create_source(
        f"url:{url}",
        "web",
        title="网页技术报告",
        authors=["Web Author"],
        year=2025,
        canonical_url=url,
        content_sha256=sha,
        status="ready",
        snapshot_path=f"{sha}.html.gz",
        snapshot_sha256=sha,
        extracted_text=text,
        locator={"kind": "web_snapshot", "snapshot_sha256": sha},
    )
    repository.add_source_identity(source.id, "url", url, is_primary=True)
    if add_content_identity:
        repository.add_source_identity(source.id, "content_sha256", sha)
    repository.add_source_record(source.id, "web", url, {"title": source.title})
    return source


def test_pdf_and_web_use_one_hybrid_search_and_evidence_pipeline(tmp_path):
    db_path = tmp_path / "documents.db"
    store = Store(db_path)
    repository = ResearchRepository(db_path)
    embedder = FakeEmbedder()

    pdf_path = make_pdf(
        tmp_path / "private-paper.pdf",
        ["hybrid evidence retrieval appears in the local PDF. " * 20],
        {"title": "本地 PDF 论文", "author": "PDF Author"},
    )
    pdf_source = repository.create_source(
        "doi:10.1000/local",
        "paper",
        title="本地 PDF 论文",
        doi="10.1000/local",
        canonical_url="https://doi.org/10.1000/local",
    )
    repository.add_source_identity(
        pdf_source.id, "doi", "10.1000/local", is_primary=True
    )
    repository.add_source_record(
        pdf_source.id, "crossref", "10.1000/local", {"DOI": "10.1000/local"}
    )
    pdf_indexed = index_pdf_source(
        store,
        repository,
        pdf_source.id,
        pdf_path,
        embedder,
        progress=noop_progress,
    )

    web_source = create_web_source(
        repository,
        text="hybrid evidence retrieval also appears in this public technical report. "
        * 20,
    )
    web_indexed = index_web_source(
        store,
        repository,
        web_source.id,
        embedder,
        progress=noop_progress,
    )

    hits = hybrid_search(store, embedder, "hybrid evidence retrieval", top=10)
    assert {hit.source_kind for hit in hits} == {"pdf", "web"}
    assert {hit.paper_id for hit in hits} == {
        pdf_indexed.paper.id,
        web_indexed.paper.id,
    }
    web_hit = next(hit for hit in hits if hit.source_kind == "web")
    assert web_hit.canonical_uri == "https://example.org/report"
    assert web_hit.path.startswith("pragent-web://")
    evidence = store.pin_evidence(store.evidence_from_chunk(web_hit.chunk_id))
    assert evidence.stale is False
    assert evidence.path.startswith("pragent-web://")
    assert "public technical report" in evidence.text

    linked_web = repository.get_source(web_source.id)
    assert linked_web.indexed_paper_id == web_indexed.paper.id
    assert linked_web.status == "ready"
    assert web_indexed.paper.locator["source_id"] == web_source.id
    assert web_indexed.paper.locator["snapshot_sha256"] == "a" * 64
    assert pdf_indexed.paper.locator == {"kind": "pdf", "source_id": pdf_source.id}
    repository.close()
    store.close()


def test_web_reindex_is_idempotent_and_source_change_makes_old_evidence_stale(tmp_path):
    db_path = tmp_path / "refresh.db"
    store = Store(db_path)
    repository = ResearchRepository(db_path)
    embedder = FakeEmbedder()
    source = create_web_source(repository, text="first snapshot evidence " * 40)

    first = index_web_source(store, repository, source.id, embedder, progress=noop_progress)
    chunk_id = store.paper_chunks(first.paper.id)[0].id
    evidence = store.pin_evidence(store.evidence_from_chunk(chunk_id))
    second = index_web_source(store, repository, source.id, embedder, progress=noop_progress)
    assert second.index_result["unchanged"] == 1
    assert second.paper.id == first.paper.id

    changed_sha = "b" * 64
    changed = repository.update_source(
        source.id,
        expected_version=repository.get_source(source.id).version,
        content_sha256=changed_sha,
        snapshot_sha256=changed_sha,
        snapshot_path=f"{changed_sha}.html.gz",
        extracted_text="second snapshot replacement evidence " * 40,
    )
    refreshed = index_web_source(
        store, repository, changed.id, embedder, progress=noop_progress
    )

    assert refreshed.index_result["updated"] == 1
    assert refreshed.paper.id == first.paper.id
    assert refreshed.paper.sha256 == changed_sha
    assert store.get_evidence(evidence.id).stale is True
    repository.close()
    store.close()


def test_content_identity_deduplicates_two_web_sources_and_removes_orphan_index(tmp_path):
    db_path = tmp_path / "dedupe.db"
    store = Store(db_path)
    repository = ResearchRepository(db_path)
    embedder = FakeEmbedder()
    first = create_web_source(repository, text="shared content " * 80)
    first_indexed = index_web_source(
        store, repository, first.id, embedder, progress=noop_progress
    )
    second = create_web_source(
        repository,
        text="shared content " * 80,
        url="https://mirror.example.org/report",
        add_content_identity=False,
    )

    second_indexed = index_web_source(
        store, repository, second.id, embedder, progress=noop_progress
    )

    assert second_indexed.deduplicated is True
    assert second_indexed.source.id == first_indexed.source.id
    assert second_indexed.paper.id == first_indexed.paper.id
    assert repository.list_sources().total == 1
    assert store.stats()[0] == 1
    assert {
        item.normalized_value
        for item in repository.list_source_identities(first_indexed.source.id)
        if item.identity_kind == "url"
    } == {
        "https://example.org/report",
        "https://mirror.example.org/report",
    }
    repository.close()
    store.close()


def test_force_pdf_rebuild_preserves_and_reembeds_web_documents(tmp_path):
    db_path = tmp_path / "force.db"
    store = Store(db_path)
    repository = ResearchRepository(db_path)
    pdf_dir = tmp_path / "library"
    pdf_dir.mkdir()
    make_pdf(pdf_dir / "main.pdf", ["main library content " * 30])
    index_library(
        store, pdf_dir, FakeEmbedder(model_name="m1"), progress=noop_progress
    )
    web = create_web_source(repository, text="preserved web evidence " * 50)
    web_indexed = index_web_source(
        store,
        repository,
        web.id,
        FakeEmbedder(model_name="m1"),
        progress=noop_progress,
    )

    index_library(
        store,
        pdf_dir,
        FakeEmbedder(model_name="m2"),
        force=True,
        progress=noop_progress,
    )

    preserved = store.paper_by_id(web_indexed.paper.id)
    assert preserved is not None and preserved.source_kind == "web"
    assert preserved.locator["source_id"] == web.id
    assert repository.get_source(web.id).indexed_paper_id == preserved.id
    hits = hybrid_search(
        store,
        FakeEmbedder(model_name="m2"),
        "preserved web evidence",
        top=10,
    )
    assert any(hit.paper_id == preserved.id for hit in hits)
    repository.close()
    store.close()


def test_document_json_uses_safe_filename_or_canonical_url_not_host_paths(tmp_path):
    db_path = tmp_path / "redaction.db"
    store = Store(db_path)
    repository = ResearchRepository(db_path)
    embedder = FakeEmbedder()
    private_dir = tmp_path / "home" / "user" / "papers"
    private_dir.mkdir(parents=True)
    pdf_path = make_pdf(
        private_dir / "secret-location.pdf",
        ["redaction searchable local text " * 20],
    )
    source = repository.create_source("record:pdf", "paper", title="Private PDF")
    repository.add_source_record(source.id, "fixture", "pdf-1", {"title": "Private PDF"})
    index_pdf_source(
        store, repository, source.id, pdf_path, embedder, progress=noop_progress
    )
    web = create_web_source(repository, text="redaction searchable web text " * 20)
    index_web_source(store, repository, web.id, embedder, progress=noop_progress)

    app = create_app(
        store=store,
        research_repository=repository,
        embedder=embedder,
        llm=OfflineLLM(),
    )
    with TestClient(app, base_url="http://127.0.0.1") as client:
        papers = client.get("/api/papers").json()["items"]
        search = client.get("/api/search", params={"q": "redaction searchable"}).json()
        status = client.get("/api/status").json()

    serialized = str({"papers": papers, "search": search, "status": status})
    assert str(private_dir) not in serialized
    assert "snapshot_path" not in serialized and "locator" not in serialized
    assert "library_dir" not in status
    pdf_item = next(item for item in papers if item["source_kind"] == "pdf")
    web_item = next(item for item in papers if item["source_kind"] == "web")
    assert pdf_item["path"] == "secret-location.pdf"
    assert web_item["path"] == "https://example.org/report"
    assert web_item["canonical_uri"] == "https://example.org/report"
    repository.close()
    store.close()
