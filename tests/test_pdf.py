from pathlib import Path

from paper_agent.pdf import extract_pdf, guess_metadata

from helpers import make_pdf


def test_extract_pdf(tmp_path):
    p = make_pdf(tmp_path / "a.pdf", ["第一页内容", "第二页内容"], {"title": "测试论文", "author": "张三; 李四"})
    pages, meta = extract_pdf(p)
    assert "第一页内容" in pages[0]
    assert "第二页内容" in pages[1]
    assert meta["title"] == "测试论文"
    assert meta["author"] == "张三; 李四"


def test_extract_pdf_broken(tmp_path):
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"not a pdf")
    try:
        extract_pdf(bad)
        assert False, "应抛出异常"
    except Exception:
        pass


def test_guess_metadata_priority(tmp_path):
    p = make_pdf(tmp_path / "whatever_name.pdf", ["正文"], {"title": "  元数据标题  ", "author": "A; B, C", "creationDate": "D:20230315..."})
    title, authors, year = guess_metadata(p, {"title": "  元数据标题  ", "author": "A; B, C", "creationDate": "D:20230315..."}, ["正文"])
    assert title == "元数据标题"
    assert authors == ["A", "B", "C"]
    assert year == 2023


def test_guess_metadata_filename_fallback(tmp_path):
    p = make_pdf(tmp_path / "my_paper_v2.pdf", ["正文"])
    title, authors, year = guess_metadata(p, {}, ["正文"])
    assert title == "my paper v2"
    assert authors == []
    assert year is None


def test_guess_metadata_year_bounds(tmp_path):
    # 未来年份与过早年份应被拒绝
    _, _, year = guess_metadata(Path("x.pdf"), {"creationDate": "D:20990101"}, ["t"])
    assert year is None
    _, _, year = guess_metadata(Path("x.pdf"), {"creationDate": "D:18000101"}, ["t"])
    assert year is None
