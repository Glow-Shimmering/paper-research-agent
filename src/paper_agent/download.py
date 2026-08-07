"""PDF 下载（arXiv）并校验。"""
import re
import urllib.request
from pathlib import Path
from typing import Optional

_ARXIV_ID_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5}(?:v[0-9]+)?)")
_UA = {"User-Agent": "paper-agent/0.2 (paper research assistant)"}


class DownloadError(Exception):
    pass


def arxiv_id_from_url(url: str) -> Optional[str]:
    m = _ARXIV_ID_RE.search(url)
    return m.group(1) if m else None


def download_pdf(url: str, target_dir: Path, timeout: int = 60) -> Path:
    """下载 arXiv PDF 到 target_dir/<arxiv_id>.pdf，校验 PDF 头后返回路径。"""
    arxiv_id = arxiv_id_from_url(url)
    if arxiv_id is None:
        raise DownloadError(f"无法从 URL 识别 arXiv 编号：{url}")
    target = target_dir / f"{arxiv_id}.pdf"
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
    try:
        req = urllib.request.Request(pdf_url, headers=_UA)
        with urllib.request.urlopen(req, timeout=timeout) as resp, open(target, "wb") as f:
            head = resp.read(5)
            if head[:4] != b"%PDF":
                raise DownloadError(
                    f"下载内容不是 PDF（Content-Type: {resp.headers.get('Content-Type')}）"
                )
            f.write(head)
            while True:
                block = resp.read(1 << 16)
                if not block:
                    break
                f.write(block)
    except DownloadError:
        raise
    except Exception as exc:
        raise DownloadError(f"下载失败：{exc}") from exc
    return target
