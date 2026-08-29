import fitz
import pytest

import pragent.download as download_module
from pragent.download import DownloadError, arxiv_id_from_url, download_pdf


def test_arxiv_id_extract():
    assert arxiv_id_from_url("https://arxiv.org/abs/2402.11651") == "2402.11651"
    assert arxiv_id_from_url("https://arxiv.org/abs/2402.11651v2") == "2402.11651v2"
    assert arxiv_id_from_url("https://arxiv.org/pdf/2301.00001v3") == "2301.00001v3"
    assert arxiv_id_from_url("http://arxiv.org/abs/2501.00001") == "2501.00001"
    assert arxiv_id_from_url("https://example.com/not-arxiv") is None
    assert arxiv_id_from_url("https://arxiv.org/abs/abc.def") is None


def valid_pdf_bytes(text: str = "valid pdf") -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    content = doc.tobytes()
    doc.close()
    return content


def fake_http_get(
    content: bytes,
    content_type: str = "application/pdf",
    content_length: int | None = None,
    status: int = 200,
):
    headers = {"content-type": content_type}
    if content_length is not None:
        headers["content-length"] = str(content_length)
    return lambda url, timeout, max_bytes: (status, headers, content)


def test_download_pdf_ok(monkeypatch, tmp_path):
    pdf_bytes = valid_pdf_bytes()
    monkeypatch.setattr(download_module, "_http_get", fake_http_get(pdf_bytes))
    target = download_pdf("https://arxiv.org/abs/2402.11651", tmp_path)
    assert target == tmp_path / "2402.11651.pdf"
    assert target.read_bytes() == pdf_bytes
    assert not (tmp_path / "2402.11651.pdf.part").exists()


def test_download_pdf_atomically_replaces_existing(monkeypatch, tmp_path):
    target = tmp_path / "2402.11651.pdf"
    target.write_bytes(b"old valid file")
    pdf_bytes = valid_pdf_bytes("replacement")
    monkeypatch.setattr(download_module, "_http_get", fake_http_get(pdf_bytes))

    returned = download_pdf("https://arxiv.org/abs/2402.11651", tmp_path)

    assert returned == target
    assert target.read_bytes() == pdf_bytes
    assert not (tmp_path / "2402.11651.pdf.part").exists()


def test_download_pdf_not_pdf(monkeypatch, tmp_path):
    target = tmp_path / "2402.11651.pdf"
    target.write_bytes(b"old pdf must survive")
    monkeypatch.setattr(
        download_module, "_http_get", fake_http_get(b"<html>not pdf</html>", "text/html")
    )
    with pytest.raises(DownloadError, match="不是 PDF"):
        download_pdf("https://arxiv.org/pdf/2402.11651", tmp_path)
    assert target.read_bytes() == b"old pdf must survive"
    assert not (tmp_path / "2402.11651.pdf.part").exists()


def test_download_pdf_bad_url(tmp_path):
    with pytest.raises(DownloadError, match="无法从 URL 识别"):
        download_pdf("https://example.com/foo", tmp_path)


def test_download_pdf_network_error(monkeypatch, tmp_path):
    target = tmp_path / "2402.11651.pdf"
    target.write_bytes(b"old pdf must survive")

    def boom(url, timeout, max_bytes):
        raise OSError("connection reset")

    monkeypatch.setattr(download_module, "_http_get", boom)
    with pytest.raises(DownloadError, match="下载失败"):
        download_pdf("https://arxiv.org/abs/2402.11651", tmp_path)
    assert target.read_bytes() == b"old pdf must survive"
    assert not (tmp_path / "2402.11651.pdf.part").exists()


def test_download_pdf_rejects_missing_target_dir(tmp_path):
    with pytest.raises(DownloadError, match="下载目录不存在"):
        download_pdf("https://arxiv.org/abs/2402.11651", tmp_path / "missing")


def test_download_pdf_size_limit_preserves_existing(monkeypatch, tmp_path):
    target = tmp_path / "2402.11651.pdf"
    target.write_bytes(b"old pdf must survive")
    monkeypatch.setattr(
        download_module, "_http_get", fake_http_get(b"%PDF" + b"x" * 100)
    )

    with pytest.raises(DownloadError, match="下载上限"):
        download_pdf("https://arxiv.org/abs/2402.11651", tmp_path, max_bytes=10)

    assert target.read_bytes() == b"old pdf must survive"
    assert list(tmp_path.glob("*.part")) == []


def test_download_pdf_rejects_truncated_content_length_and_preserves_existing(
    monkeypatch, tmp_path
):
    target = tmp_path / "2402.11651.pdf"
    target.write_bytes(b"old pdf must survive")
    complete = valid_pdf_bytes()
    truncated = complete[:-100]
    monkeypatch.setattr(
        download_module,
        "_http_get",
        fake_http_get(truncated, content_length=len(complete)),
    )

    with pytest.raises(DownloadError, match="下载不完整"):
        download_pdf("https://arxiv.org/abs/2402.11651", tmp_path)

    assert target.read_bytes() == b"old pdf must survive"
    assert list(tmp_path.glob("*.part")) == []


def test_download_pdf_rejects_corrupt_pdf_without_length(monkeypatch, tmp_path):
    target = tmp_path / "2402.11651.pdf"
    target.write_bytes(b"old pdf must survive")
    monkeypatch.setattr(
        download_module,
        "_http_get",
        fake_http_get(b"%PDF-1.7\ncorrupt body without trailer"),
    )

    with pytest.raises(DownloadError, match="不完整"):
        download_pdf("https://arxiv.org/abs/2402.11651", tmp_path)

    assert target.read_bytes() == b"old pdf must survive"


def test_http_get_follows_only_arxiv_host_redirects(monkeypatch):
    """SSRF 合同：重定向仅允许 arXiv 官方主机，且最终 2xx 才返回。"""
    import pragent.ingestion.safe_fetch as safe_fetch_mod

    calls = []

    class FakeResponse:
        def __init__(self, status, headers, body=b""):
            self.status = status
            self.headers = headers
            self.body = body

    def fake_pinned_get(url, *, headers, timeout, max_bytes, allowed_hosts=None):
        calls.append((url, allowed_hosts))
        if url.endswith("/2402.11651"):
            return FakeResponse(
                302, {"location": "https://arxiv.org/pdf/2402.11651v2"}
            )
        return FakeResponse(200, {"content-type": "application/pdf"}, b"%PDF-1.7 ok")

    monkeypatch.setattr(safe_fetch_mod, "pinned_get", fake_pinned_get)
    status, headers, body = download_module._http_get(
        "https://arxiv.org/pdf/2402.11651", timeout=10, max_bytes=1024
    )
    assert status == 200
    assert body == b"%PDF-1.7 ok"
    assert calls[0][1] is not None and "arxiv.org" in calls[0][1]
    assert len(calls) == 2


def test_http_get_redirect_loop_is_bounded(monkeypatch):
    import pragent.ingestion.safe_fetch as safe_fetch_mod

    class FakeResponse:
        def __init__(self, status, headers):
            self.status = status
            self.headers = headers
            self.body = b""

    def always_redirect(url, **kwargs):
        return FakeResponse(302, {"location": "https://arxiv.org/pdf/next"})

    monkeypatch.setattr(safe_fetch_mod, "pinned_get", always_redirect)
    with pytest.raises(DownloadError, match="超过限制"):
        download_module._http_get(
            "https://arxiv.org/pdf/2402.11651", timeout=10, max_bytes=1024
        )
