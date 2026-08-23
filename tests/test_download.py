import io
import urllib.request

import fitz
import pytest

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


def fake_urlopen(
    content: bytes,
    content_type: str = "application/pdf",
    content_length: int | None = None,
):
    class FakeResp:
        def __init__(self):
            self._buf = io.BytesIO(content)
            self.headers = {"Content-Type": content_type}
            if content_length is not None:
                self.headers["Content-Length"] = str(content_length)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, n=-1):
            return self._buf.read() if n < 0 else self._buf.read(n)

    return lambda req, timeout=None: FakeResp()


def test_download_pdf_ok(monkeypatch, tmp_path):
    pdf_bytes = valid_pdf_bytes()
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen(pdf_bytes))
    target = download_pdf("https://arxiv.org/abs/2402.11651", tmp_path)
    assert target == tmp_path / "2402.11651.pdf"
    assert target.read_bytes() == pdf_bytes
    assert not (tmp_path / "2402.11651.pdf.part").exists()


def test_download_pdf_atomically_replaces_existing(monkeypatch, tmp_path):
    target = tmp_path / "2402.11651.pdf"
    target.write_bytes(b"old valid file")
    pdf_bytes = valid_pdf_bytes("replacement")
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen(pdf_bytes))

    returned = download_pdf("https://arxiv.org/abs/2402.11651", tmp_path)

    assert returned == target
    assert target.read_bytes() == pdf_bytes
    assert not (tmp_path / "2402.11651.pdf.part").exists()


def test_download_pdf_not_pdf(monkeypatch, tmp_path):
    target = tmp_path / "2402.11651.pdf"
    target.write_bytes(b"old pdf must survive")
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen(b"<html>not pdf</html>", "text/html"))
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

    def boom(req, timeout=None):
        raise OSError("connection reset")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
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
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen(b"%PDF" + b"x" * 100))

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
        urllib.request,
        "urlopen",
        fake_urlopen(truncated, content_length=len(complete)),
    )

    with pytest.raises(DownloadError, match="下载不完整"):
        download_pdf("https://arxiv.org/abs/2402.11651", tmp_path)

    assert target.read_bytes() == b"old pdf must survive"
    assert list(tmp_path.glob("*.part")) == []


def test_download_pdf_rejects_corrupt_pdf_without_length(monkeypatch, tmp_path):
    target = tmp_path / "2402.11651.pdf"
    target.write_bytes(b"old pdf must survive")
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        fake_urlopen(b"%PDF-1.7\ncorrupt body without trailer"),
    )

    with pytest.raises(DownloadError, match="不完整"):
        download_pdf("https://arxiv.org/abs/2402.11651", tmp_path)

    assert target.read_bytes() == b"old pdf must survive"
