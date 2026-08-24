import urllib.parse

import pytest

from pragent.sources.arxiv import ArxivAdapter, parse_arxiv_feed
from pragent.sources.base import SourceProvider, SourceProviderError
from pragent.sources.crossref import CrossrefAdapter
from pragent.sources.http import HttpResponse, JsonHttpClient, RateLimiter
from pragent.sources.semantic_scholar import SemanticScholarAdapter

ARXIV_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>https://arxiv.org/abs/2402.11651v2</id>
    <updated>2024-02-20T00:00:00Z</updated>
    <published>2024-02-18T00:00:00Z</published>
    <title>  Retrieval Augmented Generation  </title>
    <summary> Evidence-grounded generation. </summary>
    <author><name>Alice</name></author>
    <author><name>Bob</name></author>
    <arxiv:doi>10.1000/RAG.TEST</arxiv:doi>
    <arxiv:journal_ref>Journal 1</arxiv:journal_ref>
    <category term="cs.IR"/>
    <link rel="alternate" href="https://arxiv.org/abs/2402.11651v2"/>
    <link rel="related" title="pdf" href="https://arxiv.org/pdf/2402.11651v2"/>
  </entry>
</feed>
"""


class NoWaitLimiter:
    def wait(self):
        pass


def test_arxiv_fixture_normalizes_contract_and_preserves_provenance():
    records = parse_arxiv_feed(ARXIV_XML)

    assert len(records) == 1
    item = records[0]
    assert item.provider == "arxiv"
    assert item.provider_record_id == "2402.11651"
    assert item.title == "Retrieval Augmented Generation"
    assert item.authors == ("Alice", "Bob")
    assert item.year == 2024
    assert item.doi == "10.1000/rag.test"
    assert item.arxiv_id == "2402.11651"
    assert item.pdf_url.endswith("2402.11651v2")
    assert item.metadata["categories"] == ["cs.IR"]
    assert item.provenance.raw_metadata["abstract"] == "Evidence-grounded generation."


def test_arxiv_adapter_search_and_lookup_use_bounded_contract():
    calls = []

    def requester(url, headers, timeout):
        calls.append((url, headers, timeout))
        return ARXIV_XML

    adapter = ArxivAdapter(
        requester=requester,
        limiter=NoWaitLimiter(),
        timeout=7,
    )
    assert isinstance(adapter, SourceProvider)

    search = adapter.search("retrieval agents", limit=3)
    looked_up = adapter.lookup("arXiv:2402.11651v9")

    assert search[0].provider_record_id == "2402.11651"
    assert looked_up == search[0]
    search_query = urllib.parse.parse_qs(urllib.parse.urlsplit(calls[0][0]).query)
    lookup_query = urllib.parse.parse_qs(urllib.parse.urlsplit(calls[1][0]).query)
    assert search_query["max_results"] == ["3"]
    assert lookup_query["id_list"] == ["2402.11651"]
    assert calls[0][1]["User-Agent"].startswith("PRAgent/")
    assert calls[0][2] == 7


def test_arxiv_adapter_maps_network_and_parse_failures():
    def network_failure(url, headers, timeout):
        raise OSError("offline")

    adapter = ArxivAdapter(requester=network_failure, limiter=NoWaitLimiter())
    with pytest.raises(SourceProviderError) as captured:
        adapter.search("query")
    assert captured.value.provider == "arxiv"
    assert captured.value.code == "network_error"
    assert captured.value.retryable is True

    malformed = ArxivAdapter(
        requester=lambda url, headers, timeout: b"not xml",
        limiter=NoWaitLimiter(),
    )
    with pytest.raises(SourceProviderError) as captured:
        malformed.search("query")
    assert captured.value.code == "invalid_response"


def test_arxiv_adapter_rejects_unbounded_or_invalid_inputs_before_request():
    adapter = ArxivAdapter(
        requester=lambda url, headers, timeout: ARXIV_XML,
        limiter=NoWaitLimiter(),
    )
    with pytest.raises(ValueError, match="query"):
        adapter.search("  ")
    with pytest.raises(ValueError, match="1–100"):
        adapter.search("x", limit=101)
    with pytest.raises(ValueError, match="arXiv"):
        adapter.lookup("not-an-id")


def test_semantic_scholar_fixture_optional_auth_and_normalization():
    payload = b'''{
      "total": 1,
      "data": [{
        "paperId": "CorpusId:123",
        "title": "Evidence First Agents",
        "authors": [{"authorId": "1", "name": "Alice"}],
        "year": 2025,
        "abstract": "Grounded research.",
        "externalIds": {"DOI": "10.1000/AGENT", "ArXiv": "2501.00001v2"},
        "url": "https://www.semanticscholar.org/paper/123",
        "openAccessPdf": {"url": "https://arxiv.org/pdf/2501.00001"}
      }]
    }'''
    calls = []

    def requester(url, headers, timeout, max_bytes):
        calls.append((url, dict(headers)))
        return HttpResponse(200, {"Content-Type": "application/json"}, payload)

    client = JsonHttpClient(
        "semantic_scholar", requester=requester, limiter=RateLimiter(0)
    )
    adapter = SemanticScholarAdapter(api_key="optional-secret", client=client)
    item = adapter.search("agents", limit=3)[0]

    assert item.provider_record_id == "CorpusId:123"
    assert item.authors == ("Alice",)
    assert item.doi == "10.1000/agent"
    assert item.arxiv_id == "2501.00001"
    assert item.pdf_url == "https://arxiv.org/pdf/2501.00001"
    assert calls[0][1]["x-api-key"] == "optional-secret"
    assert "optional-secret" not in calls[0][0]


def test_crossref_fixture_polite_header_and_normalization():
    payload = b'''{
      "status": "ok",
      "message": {"items": [{
        "DOI": "10.5555/RAG.1",
        "title": ["Retrieval Systems"],
        "author": [{"given": "Alice", "family": "Chen"}],
        "published-online": {"date-parts": [[2024, 7, 1]]},
        "URL": "https://doi.org/10.5555/RAG.1",
        "abstract": "<jats:p>Evidence &amp; retrieval.</jats:p>",
        "link": [{"content-type": "application/pdf", "URL": "https://example.org/paper.pdf"}]
      }]}
    }'''
    calls = []

    def requester(url, headers, timeout, max_bytes):
        calls.append((url, dict(headers)))
        return HttpResponse(200, {}, payload)

    adapter = CrossrefAdapter(
        contact_email="researcher@example.com",
        client=JsonHttpClient("crossref", requester=requester, limiter=RateLimiter(0)),
    )
    item = adapter.search("retrieval", limit=2)[0]

    assert item.provider_record_id == "10.5555/rag.1"
    assert item.title == "Retrieval Systems"
    assert item.authors == ("Alice Chen",)
    assert item.year == 2024
    assert item.abstract == "Evidence & retrieval."
    assert item.pdf_url == "https://example.org/paper.pdf"
    assert "mailto:researcher@example.com" in calls[0][1]["User-Agent"]
    assert "query.bibliographic=retrieval" in calls[0][0]


def test_semantic_and_crossref_reject_malformed_fixture_shapes():
    semantic = SemanticScholarAdapter(
        client=JsonHttpClient(
            "semantic_scholar",
            requester=lambda *args: HttpResponse(200, {}, b'{"unexpected":[]}'),
            limiter=RateLimiter(0),
        )
    )
    with pytest.raises(SourceProviderError, match="data"):
        semantic.search("x")

    crossref = CrossrefAdapter(
        client=JsonHttpClient(
            "crossref",
            requester=lambda *args: HttpResponse(200, {}, b'{"message":{}}'),
            limiter=RateLimiter(0),
        )
    )
    with pytest.raises(SourceProviderError, match="items"):
        crossref.search("x")
