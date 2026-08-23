import io
import time
import urllib.request

import pytest

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


def fake_urlopen(body: bytes):
    class FakeResp:
        def __init__(self):
            self._buf = io.BytesIO(body)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return self._buf.read()

    return lambda url, timeout: FakeResp()


def test_search_papers_parse(monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen(SAMPLE_XML.encode()))
    monkeypatch.setattr(time, "sleep", lambda s: None)
    papers = search_papers("attention", limit=5)
    assert len(papers) == 1
    p = papers[0]
    assert p.title == "Attention Is All You Need"
    assert p.authors == ["Alice Chen", "Bob Wang"]
    assert p.year == 2023
    assert p.url == "http://arxiv.org/abs/2301.00001v1"
    assert p.pdf_url == "http://arxiv.org/pdf/2301.00001v1"
    assert "new architecture" in p.abstract


def test_search_papers_empty(monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen(EMPTY_XML.encode()))
    monkeypatch.setattr(time, "sleep", lambda s: None)
    assert search_papers("zzz-not-found-xyz", limit=5) == []


def test_search_papers_network_error(monkeypatch):
    def boom(url, timeout):
        raise OSError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    monkeypatch.setattr(time, "sleep", lambda s: None)
    with pytest.raises(WebSearchError, match="arXiv 请求失败"):
        search_papers("attention")


def test_search_papers_bad_xml(monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen(b"not xml at all"))
    monkeypatch.setattr(time, "sleep", lambda s: None)
    with pytest.raises(WebSearchError, match="解析失败"):
        search_papers("attention")
