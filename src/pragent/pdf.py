"""PDF 文本与元数据提取。"""
import re
from datetime import date
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF

_YEAR_RE = re.compile(r"(\d{4})")


def extract_pdf(path: str | Path) -> tuple[list[str], dict]:
    """返回 (逐页文本列表, 内嵌 metadata 字典)。解密失败/损坏时抛异常。"""
    with fitz.open(str(path)) as doc:
        pages = [page.get_text("text") for page in doc]
        meta = dict(doc.metadata or {})
    return pages, meta


def guess_metadata(path: Path, meta: dict, pages: list[str]) -> tuple[str, list[str], Optional[int]]:
    """按优先级推断 (title, authors, year)：PDF metadata > 文件名兜底。"""
    title = (meta.get("title") or "").strip()
    if not title:
        title = path.stem.replace("_", " ").replace("-", " ").strip() or path.stem

    authors: list[str] = []
    raw_author = (meta.get("author") or "").strip()
    for part in raw_author.replace(";", ",").split(","):
        part = part.strip()
        if part:
            authors.append(part)

    year: Optional[int] = None
    this_year = date.today().year
    for key in ("creationDate", "modDate"):
        m = _YEAR_RE.search(meta.get(key) or "")
        if m:
            y = int(m.group(1))
            if 1900 <= y <= this_year + 1:
                year = y
                break

    return title, authors, year
