"""PDF 下载（arXiv）并校验。"""
import os
import re
import tempfile
import urllib.parse
from pathlib import Path
from typing import Optional

import fitz

_ARXIV_ID_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5}(?:v[0-9]+)?)")
# 下载目标文件名与请求 URL 都由该编号构造；在构造点再次全匹配校验，
# 显式排除路径分隔符等字符（即使上游正则被修改也保持安全）。
_ARXIV_ID_SAFE_RE = re.compile(r"[0-9]{4}\.[0-9]{4,5}(?:v[0-9]+)?")
_UA = {"User-Agent": "PRAgent/0.1 (paper research assistant)"}
_MAX_DOWNLOAD_BYTES = 200 * 1024 * 1024
_DOWNLOAD_ALLOWED_HOSTS = frozenset({"arxiv.org", "www.arxiv.org", "export.arxiv.org"})
_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
_MAX_REDIRECTS = 5


class DownloadError(Exception):
    pass


def _http_get(url: str, timeout: float, max_bytes: int):
    """SSRF-safe GET：逐跳校验允许主机与公网 IP，返回 (status, headers, body)。

    重定向仅允许在 arXiv 官方主机之间进行，且每一跳都重新解析并固定
    公网 IP；响应整体受 ``max_bytes`` 硬限制。
    """
    from .ingestion.safe_fetch import SafeFetchError, pinned_get

    current = url
    for _ in range(_MAX_REDIRECTS + 1):
        try:
            response = pinned_get(
                current,
                headers=_UA,
                timeout=timeout,
                max_bytes=max_bytes,
                allowed_hosts=_DOWNLOAD_ALLOWED_HOSTS,
            )
        except SafeFetchError as exc:
            if exc.code == "response_too_large":
                raise DownloadError(
                    f"PDF 超过下载上限（{max_bytes // (1024 * 1024)}MB）"
                ) from exc
            raise DownloadError(f"下载失败：{exc}") from exc
        if response.status in _REDIRECT_STATUSES:
            location = (response.headers.get("location") or "").strip()
            if not location:
                raise DownloadError("下载重定向缺少 Location")
            current = urllib.parse.urljoin(current, location)
            continue
        headers = {str(key): str(value) for key, value in response.headers.items()}
        return int(response.status), headers, response.body
    raise DownloadError("下载重定向次数超过限制")


def _validate_complete_pdf(path: Path) -> None:
    """拒绝只有 PDF 文件头、缺少结尾或无法完整解析的下载。"""
    try:
        size = path.stat().st_size
        with path.open("rb") as file:
            file.seek(max(0, size - 4096))
            tail = file.read()
        if b"%%EOF" not in tail:
            raise DownloadError("下载的 PDF 不完整（缺少 EOF 标记）")
        with fitz.open(path) as document:
            if not document.is_pdf or document.page_count < 1:
                raise DownloadError("下载内容不是可用的 PDF")
            # 同时加载首尾页，避免仅目录可读而页面对象已截断/损坏。
            document.load_page(0)
            document.load_page(document.page_count - 1)
    except DownloadError:
        raise
    except Exception as exc:
        raise DownloadError(f"下载的 PDF 无法完整解析：{exc}") from exc


def arxiv_id_from_url(url: str) -> Optional[str]:
    m = _ARXIV_ID_RE.search(url)
    return m.group(1) if m else None


def download_pdf(
    url: str,
    target_dir: Path,
    timeout: float = 60,
    max_bytes: int = _MAX_DOWNLOAD_BYTES,
) -> Path:
    """下载到临时 .part，校验后原子替换目标；失败时保留旧 PDF。

    ``timeout`` 为本次下载的网络预算（秒），由调用方按剩余执行预算传入。
    """
    arxiv_id = arxiv_id_from_url(url)
    if arxiv_id is None or not _ARXIV_ID_SAFE_RE.fullmatch(arxiv_id):
        raise DownloadError(f"无法从 URL 识别 arXiv 编号：{url}")
    try:
        target_dir = target_dir.expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise DownloadError(f"下载目录不存在或无法访问：{target_dir}") from exc
    if not target_dir.is_dir():
        raise DownloadError(f"下载路径不是文件夹：{target_dir}")
    # arxiv_id 已通过 _ARXIV_ID_SAFE_RE 全匹配校验；再经 basename 归一化
    # 剥离任何路径分隔符残留，保证目标不会逃出下载目录。
    filename = os.path.basename(arxiv_id + ".pdf")
    if filename != arxiv_id + ".pdf" or filename in ("", ".", ".."):
        raise DownloadError("下载目标文件名不安全")
    target = target_dir / filename
    fd, raw_partial = tempfile.mkstemp(
        prefix=f".{arxiv_id}.", suffix=".pdf.part", dir=target_dir
    )
    os.close(fd)
    partial = Path(raw_partial)
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
    try:
        status, headers, body = _http_get(pdf_url, timeout, max_bytes)
        if status != 200:
            raise DownloadError(f"下载失败：HTTP {status}")
        raw_length = headers.get("content-length")
        expected_length = None
        if raw_length:
            try:
                expected_length = int(raw_length)
                if expected_length < 0:
                    raise ValueError
                if expected_length > max_bytes:
                    raise DownloadError(f"PDF 超过下载上限（{max_bytes // (1024 * 1024)}MB）")
            except ValueError as exc:
                raise DownloadError("下载响应的 Content-Length 无效") from exc
        if len(body) > max_bytes:
            raise DownloadError(f"PDF 超过下载上限（{max_bytes // (1024 * 1024)}MB）")
        if body[:4] != b"%PDF":
            raise DownloadError(
                f"下载内容不是 PDF（Content-Type: {headers.get('content-type')}）"
            )
        partial.write_bytes(body)
        if expected_length is not None and len(body) != expected_length:
            raise DownloadError(
                f"PDF 下载不完整（预期 {expected_length} 字节，实际 {len(body)} 字节）"
            )
        _validate_complete_pdf(partial)
        os.replace(partial, target)
    except DownloadError:
        raise
    except Exception as exc:
        raise DownloadError(f"下载失败：{exc}") from exc
    finally:
        # 成功 replace 后临时文件已不存在；失败则只清理临时文件，绝不碰旧目标。
        try:
            partial.unlink(missing_ok=True)
        except OSError:
            pass
    return target
