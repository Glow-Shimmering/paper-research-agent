import sqlite3

import pytest

from pragent.ingestion.html_extract import HtmlExtractionError, extract_html
from pragent.ingestion.safe_fetch import SafeFetchError, SafeFetchResult
from pragent.ingestion.snapshots import SnapshotStore
from pragent.ingestion.web import WebIngestService
from pragent.storage import ResearchRepository
from pragent.store import Store

ARTICLE = b'''<!doctype html>
<html lang="en">
<head>
  <title>Fallback title</title>
  <meta name="citation_title" content="Evidence First Research Agents">
  <meta name="citation_author" content="Alice Chen">
  <meta name="citation_date" content="2024-05-01">
</head>
<body>
  <nav>private navigation should not be extracted</nav>
  <article>
    <h1>Evidence First Research Agents</h1>
    <p>Retrieval augmented agents ground every claim in verifiable source passages.</p>
    <p>The evaluation compares citation precision and evidence recall across systems.</p>
    <p>Limitations include incomplete metadata and unavailable full text documents.</p>
  </article>
  <script>window.secret = 'never execute or extract';</script>
</body>
</html>'''


class FixtureFetcher:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = []

    def fetch(self, url):
        self.calls.append(url)
        if self.error:
            raise self.error
        return self.result


def fetched(body=ARTICLE):
    return SafeFetchResult(
        requested_url="https://example.org/start",
        final_url="https://www.example.org/report",
        status_code=200,
        content_type="text/html",
        body=body,
        redirect_chain=("https://www.example.org/report",),
        resolved_ips=("93.184.216.34", "93.184.216.35"),
    )


def test_trafilatura_extracts_text_metadata_without_raw_html():
    document = extract_html(ARTICLE, final_url="https://www.example.org/report")

    assert document.title == "Evidence First Research Agents"
    assert document.authors == ("Alice Chen",)
    assert document.year == 2024
    assert "Retrieval augmented agents" in document.text
    assert "private navigation" not in document.text
    assert "window.secret" not in document.text
    assert document.canonical_url == "https://www.example.org/report"
    assert len(document.text_sha256) == 64
    assert "<article>" not in str(document.metadata)


def test_web_ingestion_persists_relative_snapshot_extracted_text_and_provenance(tmp_path):
    db_path = tmp_path / "web.db"
    Store(db_path).close()
    repository = ResearchRepository(db_path)
    snapshots = SnapshotStore(tmp_path / "private-snapshots")
    service = WebIngestService(
        repository,
        fetcher=FixtureFetcher(fetched()),
        snapshots=snapshots,
    )

    result = service.ingest("https://example.org/start#fragment")

    source = result.source
    assert source.status == "ready"
    assert source.source_kind == "web"
    assert source.title == "Evidence First Research Agents"
    assert source.canonical_url == "https://www.example.org/report"
    assert source.snapshot_path == result.snapshot.relative_path
    assert not source.snapshot_path.startswith(str(tmp_path))
    assert source.snapshot_sha256 == result.snapshot.sha256
    assert "Retrieval augmented agents" in source.extracted_text
    assert source.locator == {
        "kind": "web_snapshot",
        "final_url": "https://www.example.org/report",
        "snapshot_sha256": result.snapshot.sha256,
    }
    assert snapshots.read(source.snapshot_path) == ARTICLE
    assert {
        (identity.identity_kind, identity.normalized_value)
        for identity in repository.list_source_identities(source.id)
    } == {
        ("url", "https://example.org/start"),
        ("url", "https://www.example.org/report"),
        ("content_sha256", result.snapshot.sha256),
    }
    records = repository.list_source_records(source.id)
    assert {record.provider_record_id for record in records} == {
        "https://example.org/start",
        "https://www.example.org/report",
    }
    assert all("<html" not in str(record.raw_metadata).lower() for record in records)

    connection = sqlite3.connect(db_path)
    raw_records = connection.execute("SELECT raw_metadata FROM source_records").fetchall()
    assert all("window.secret" not in row[0] for row in raw_records)
    connection.close()
    repository.close()


def test_web_ingestion_failure_is_persisted_and_same_url_can_recover(tmp_path):
    db_path = tmp_path / "retry.db"
    Store(db_path).close()
    repository = ResearchRepository(db_path)
    snapshots = SnapshotStore(tmp_path / "snapshots")
    failure = SafeFetchError(
        "fixture timeout", code="timeout", retryable=True
    )
    failing = WebIngestService(
        repository,
        fetcher=FixtureFetcher(error=failure),
        snapshots=snapshots,
    )

    with pytest.raises(SafeFetchError, match="timeout"):
        failing.ingest("https://example.org/article")
    failed = repository.list_sources().items[0]
    assert failed.status == "failed"
    assert failed.metadata["last_error"]["code"] == "timeout"

    recovered = WebIngestService(
        repository,
        fetcher=FixtureFetcher(
            SafeFetchResult(
                requested_url="https://example.org/article",
                final_url="https://example.org/article",
                status_code=200,
                content_type="text/html",
                body=ARTICLE,
                redirect_chain=(),
                resolved_ips=("93.184.216.34",),
            )
        ),
        snapshots=snapshots,
    ).ingest("https://example.org/article")
    assert recovered.source.id == failed.id
    assert recovered.source.status == "ready"
    assert repository.list_sources().total == 1
    repository.close()


def test_extraction_failure_marks_source_failed_but_never_stores_raw_html_in_db(tmp_path):
    db_path = tmp_path / "empty.db"
    Store(db_path).close()
    repository = ResearchRepository(db_path)
    body = b"<html><head><title>Empty</title></head><body></body></html>"
    service = WebIngestService(
        repository,
        fetcher=FixtureFetcher(fetched(body)),
        snapshots=SnapshotStore(tmp_path / "snapshots"),
    )

    with pytest.raises(HtmlExtractionError):
        service.ingest("https://example.org/start")
    source = repository.list_sources().items[0]
    assert source.status == "failed"
    assert source.extracted_text is None
    connection = sqlite3.connect(db_path)
    assert "<html" not in connection.execute(
        "SELECT raw_metadata FROM source_records"
    ).fetchone()[0].lower()
    connection.close()
    repository.close()
