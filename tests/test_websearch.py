import pytest

import pragent.websearch as websearch_module
from pragent.sources.arxiv import ArxivAdapter
from pragent.sources.base import SourceProviderError
from pragent.sources.http import RateLimiter
from pragent.websearch import WebSearchError, search_papers

SAMPLE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2301.00001v1</id>
    <title>   Attention Is All You Need   </title>
    <summary>
      We propose a new architecture based on attention.
    </summary>
    <published>2023-01-15T00:00:00Z</published>
    <author><name>Alice Chen</name></author>
    <author><name>Bob Wang</name></author>
    <link rel="alternate" href="http://arxiv.org/abs/2301.00001v1"/>
    <link rel="related" title="pdf" href="http://arxiv.org/pdf/2301.00001v1"/>
  </entry>
</feed>
"""

EMPTY_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
</feed>
"""


def install_offline_adapter(monkeypatch, body_or_error):
    """用注入 requester 的离线 adapter 替换 websearch 的模块级单例。"""

    calls = []

    def requester(url, headers, timeout):
        calls.append((url, dict(headers), timeout))
        if isinstance(body_or_error, Exception):
            raise body_or_error
        return body_or_error

    monkeypatch.setattr(
        websearch_module,
        "_ADAPTER",
        ArxivAdapter(requester=requester, limiter=RateLimiter(0)),
    )
    return calls


def test_search_papers_parse(monkeypatch):
    calls = install_offline_adapter(monkeypatch, SAMPLE_XML.encode())
    papers = search_papers("attention", limit=5)
    assert len(papers) == 1
    p = papers[0]
    assert p.title == "Attention Is All You Need"
    assert p.authors == ["Alice Chen", "Bob Wang"]
    assert p.year == 2023
    assert p.url == "http://arxiv.org/abs/2301.00001v1"
    assert p.pdf_url == "http://arxiv.org/pdf/2301.00001v1"
    assert "new architecture" in p.abstract
    # provider 契约：search 请求打在官方 API 主机上。
    assert calls[0][0].startswith("https://export.arxiv.org/api/query")


def test_search_papers_forwards_timeout_budget(monkeypatch):
    calls = install_offline_adapter(monkeypatch, EMPTY_XML.encode())
    search_papers("attention", limit=5, timeout=4.5)
    assert calls[0][2] == 4.5


def test_search_papers_empty(monkeypatch):
    install_offline_adapter(monkeypatch, EMPTY_XML.encode())
    assert search_papers("zzz-not-found-xyz", limit=5) == []


def test_search_papers_network_error(monkeypatch):
    install_offline_adapter(monkeypatch, SourceProviderError("down", provider="arxiv"))
    with pytest.raises(WebSearchError, match="down"):
        search_papers("attention")


def test_search_papers_bad_xml(monkeypatch):
    install_offline_adapter(monkeypatch, b"not xml at all")
    with pytest.raises(WebSearchError, match="解析失败"):
        search_papers("attention")


def test_request_bytes_rejects_non_arxiv_host():
    """SSRF 合同：_request_bytes 只允许 arXiv 官方主机（解析前即拒绝）。"""
    import pragent.sources.arxiv as arxiv_mod
    from pragent.ingestion.safe_fetch import SafeFetchError

    with pytest.raises(SafeFetchError, match="主机不在允许列表"):
        arxiv_mod._request_bytes(
            "https://evil.example.org/api/query", {}, timeout=1.0
        )


def test_parse_arxiv_feed_rejects_dtd():
    """XML 实体扩展合同：包含 DTD/实体声明的响应直接拒绝。"""
    from pragent.sources.arxiv import parse_arxiv_feed

    malicious = b"""<?xml version="1.0"?>
<!DOCTYPE feed [<!ENTITY xxe SYSTEM "file:///etc/passwd">]>
<feed xmlns="http://www.w3.org/2005/Atom"></feed>
"""
    with pytest.raises(SourceProviderError, match="DTD"):
        parse_arxiv_feed(malicious)
