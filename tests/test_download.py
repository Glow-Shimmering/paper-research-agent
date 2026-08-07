import io
import urllib.request

import pytest

from paper_agent.download import DownloadError, arxiv_id_from_url, download_pdf


def test_arxiv_id_extract():
    assert arxiv_id_from_url("https://arxiv.org/abs/2402.11651") == "2402.11651"
    assert arxiv_id_from_url("https://arxiv.org/abs/2402.11651v2") == "2402.11651v2"
    assert arxiv_id_from_url("https://arxiv.org/pdf/2301.00001v3") == "2301.00001v3"
    assert arxiv_id_from_url("http://arxiv.org/abs/2501.00001") == "2501.00001"
    assert arxiv_id_from_url("https://example.com/not-arxiv") is None
    assert arxiv_id_from_url("https://arxiv.org/abs/abc.def") is None


def fake_urlopen(content: bytes, content_type: str = "application/pdf"):
    class FakeResp:
        headers = {"Content-Type": content_type}

        def __init__(self):
            self._buf = io.BytesIO(content)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, n=-1):
            return self._buf.read() if n < 0 else self._buf.read(n)

    return lambda req, timeout=None: FakeResp()


def test_download_pdf_ok(monkeypatch, tmp_path):
    pdf_bytes = b"%PDF-1.7 fake content " * 100
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen(pdf_bytes))
    target = download_pdf("https://arxiv.org/abs/2402.11651", tmp_path)
    assert target == tmp_path / "2402.11651.pdf"
    assert target.read_bytes() == pdf_bytes


def test_download_pdf_not_pdf(monkeypatch, tmp_path):
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen(b"<html>not pdf</html>", "text/html"))
    with pytest.raises(DownloadError, match="不是 PDF"):
        download_pdf("https://arxiv.org/pdf/2402.11651", tmp_path)


def test_download_pdf_bad_url(tmp_path):
    with pytest.raises(DownloadError, match="无法从 URL 识别"):
        download_pdf("https://example.com/foo", tmp_path)


def test_download_pdf_network_error(monkeypatch, tmp_path):
    def boom(req, timeout=None):
        raise OSError("connection reset")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    with pytest.raises(DownloadError, match="下载失败"):
        download_pdf("https://arxiv.org/abs/2402.11651", tmp_path)
