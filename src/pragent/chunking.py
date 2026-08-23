"""分块：段落优先，目标 800 字符、重叠 120 字符，重叠处取句子边界。"""
import re

_SENT_BOUNDARY = re.compile(r"[。！？!?\n]")


def chunk_text(pages: list[str], target: int = 800, overlap: int = 120) -> list[tuple[int, str]]:
    """按 (页码, 文本) 分块。纯函数、确定性。空输入返回 []。"""
    if not pages or all(not p.strip() for p in pages):
        return []

    # 段落 = 按空行切分，记录段落首字符所在页码
    paragraphs: list[tuple[int, str]] = []
    for page_idx, page_text in enumerate(pages, start=1):
        for para in page_text.split("\n\n"):
            para = para.strip()
            if para:
                paragraphs.append((page_idx, para))

    chunks: list[tuple[int, str]] = []
    buf: list[str] = []
    buf_len = 0
    buf_page = paragraphs[0][0]

    def flush() -> None:
        nonlocal buf, buf_len
        if buf:
            chunks.append((buf_page, "".join(buf)))
        buf = []
        buf_len = 0

    for page, para in paragraphs:
        if buf_len > 0 and buf_len + len(para) > target:
            flush()
            buf_page = page
        if len(para) <= target:
            buf.append(para)
            buf.append("\n")
            buf_len += len(para) + 1
        else:
            # 超长段落：在句子边界切成 ≤target 的片
            pieces = _split_long(para, target)
            for piece in pieces:
                if buf_len > 0 and buf_len + len(piece) > target:
                    flush()
                    buf_page = page
                buf.append(piece)
                buf.append("\n")
                buf_len += len(piece) + 1
    flush()

    # 重叠：每块（除第一块）开头并入上一块末尾 ≤overlap 字符，取句子边界
    result: list[tuple[int, str]] = []
    for idx, (page, text) in enumerate(chunks):
        if idx == 0:
            result.append((page, text))
            continue
        prev_text = result[-1][1]
        window = prev_text[-overlap:]
        cut = _last_sentence_boundary(window)
        prefix = prev_text[len(prev_text) - overlap + cut:] if cut >= 0 else prev_text[-overlap:]
        result.append((page, prefix + text))
    return result


def _split_long(para: str, target: int) -> list[str]:
    pieces: list[str] = []
    rest = para
    while len(rest) > target:
        window = rest[:target]
        cut = _last_sentence_boundary(window)
        split_at = cut + 1 if cut >= 0 else target
        pieces.append(rest[:split_at])
        rest = rest[split_at:]
    if rest:
        pieces.append(rest)
    return pieces


def _last_sentence_boundary(text: str) -> int:
    """返回 text 中最后一个有后继字符的句子边界位置；无则 -1。"""
    for m in reversed(list(_SENT_BOUNDARY.finditer(text))):
        if m.start() < len(text) - 1:
            return m.start()
    return -1
