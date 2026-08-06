from paper_agent.chunking import chunk_text


def test_empty_input():
    assert chunk_text([]) == []
    assert chunk_text(["", "  "]) == []


def test_small_text_single_chunk():
    chunks = chunk_text(["第一段。第二段。"])
    assert len(chunks) == 1
    assert chunks[0][1] == "第一段。第二段。\n"
    assert chunks[0][0] == 1


def test_chunk_size_limit():
    pages = ["甲" * 3000]
    chunks = chunk_text(pages, target=800, overlap=120)
    assert len(chunks) >= 3
    for page, text in chunks:
        assert page == 1
        # 硬切块 800 + 换行；重叠前缀最多 120
        assert len(text) <= 800 + 120 + 1


def test_overlap_from_previous_chunk():
    text = "".join(f"第{i}段内容。" for i in range(50))
    chunks = chunk_text([text], target=200, overlap=40)
    assert len(chunks) > 1
    # 每块（除首块）以某段非空后缀（≤overlap 字符）开头
    suffixes = [prev[1][-i:] for prev in chunks[:-1] for i in range(1, 41)]
    for cur in chunks[1:]:
        assert any(cur[1].startswith(s) for s in suffixes)


def test_page_mapping():
    pages = [
        "第一页" + "这是段落内容。" * 40,
        "第二页" + "这是段落内容。" * 40,
    ]
    chunks = chunk_text(pages, target=200, overlap=40)
    pages_seen = {p for p, _ in chunks}
    assert pages_seen == {1, 2}
    seq = [p for p, _ in chunks]
    assert seq == sorted(seq)


def test_long_paragraph_sentence_split():
    text = "短句。" * 300  # 300 个"短句。"，每句 3 字符
    chunks = chunk_text([text], target=100, overlap=30)
    assert len(chunks) > 2
    for _, c in chunks:
        assert len(c) <= 100 + 30 + 1
