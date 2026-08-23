"""PDF 下载（arXiv）并校验。"""
import os
import re
import tempfile
import urllib.request
from pathlib import Path
from typing import Optional

import fitz

_ARXIV_ID_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5}(?:v[0-9]+)?)")
_UA = {"User-Agent": "PRAgent/0.1 (paper research assistant)"}
_MAX_DOWNLOAD_BYTES = 200 * 1024 * 1024


class DownloadError(Exception):
    pass


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
    timeout: int = 60,
    max_bytes: int = _MAX_DOWNLOAD_BYTES,
) -> Path:
    """下载到临时 .part，校验后原子替换目标；失败时保留旧 PDF。"""
    arxiv_id = arxiv_id_from_url(url)
    if arxiv_id is None:
        raise DownloadError(f"无法从 URL 识别 arXiv 编号：{url}")
    try:
        target_dir = target_dir.expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise DownloadError(f"下载目录不存在或无法访问：{target_dir}") from exc
    if not target_dir.is_dir():
        raise DownloadError(f"下载路径不是文件夹：{target_dir}")
    target = target_dir / f"{arxiv_id}.pdf"
    fd, raw_partial = tempfile.mkstemp(
        prefix=f".{arxiv_id}.", suffix=".pdf.part", dir=target_dir
    )
    os.close(fd)
    partial = Path(raw_partial)
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
    try:
        req = urllib.request.Request(pdf_url, headers=_UA)
        with urllib.request.urlopen(req, timeout=timeout) as resp, open(partial, "wb") as f:
            raw_length = resp.headers.get("Content-Length")
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
            head = resp.read(5)
            if head[:4] != b"%PDF":
                raise DownloadError(
                    f"下载内容不是 PDF（Content-Type: {resp.headers.get('Content-Type')}）"
                )
            f.write(head)
            written = len(head)
            while True:
                block = resp.read(1 << 16)
                if not block:
                    break
                written += len(block)
                if written > max_bytes:
                    raise DownloadError(f"PDF 超过下载上限（{max_bytes // (1024 * 1024)}MB）")
                f.write(block)
        if expected_length is not None and written != expected_length:
            raise DownloadError(
                f"PDF 下载不完整（预期 {expected_length} 字节，实际 {written} 字节）"
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
