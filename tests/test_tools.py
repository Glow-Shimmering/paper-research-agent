import hashlib
import json
import logging

import pytest

from pragent.store import Store
from pragent.storage.research_repository import ResearchRepository
from pragent.tool_protocol import ToolEffect, ToolResult, ToolSpec, ToolValidationError
from pragent.tools import (
    CONFIRMATION_TOOLS,
    EXTERNAL_TOOLS,
    MUTATING_TOOLS,
    TOOLS,
    ToolContext,
    execute_tool,
    execute_tool_result,
    register_tool,
    unregister_tool,
)

from helpers import FakeEmbedder, make_paper


class FakeLLM:
    is_configured = True


def make_ctx(tmp_path, library_dir=None):
    s = Store(tmp_path / "t.db")
    pid = s.upsert_paper(make_paper("a.pdf", title="注意力机制研究", authors=["张三"], year=2023))
    from pragent.models import Chunk
    from helpers import FakeEmbedder as FE

    s.replace_chunks(
        pid,
        [
            Chunk(None, pid, 0, 1, "注意力机制是文本分类的关键技术。", FE.vecs_for("注意力机制是文本分类的关键技术。")),
            Chunk(None, pid, 1, 2, "Transformer 使用自注意力。", FE.vecs_for("Transformer 使用自注意力。")),
        ],
    )
    if library_dir is not None:
        s.meta_set("library_dir", str(library_dir))
    return ToolContext(
        store=s,
        embedder=FakeEmbedder(),
        llm=FakeLLM(),
        require_confirmation=False,
    )


def test_tools_schema_complete():
    from pragent.tools import SCHEMA_NAMES

    names = [t["function"]["name"] for t in TOOLS]
    assert set(names) == SCHEMA_NAMES == {
        "local_search", "web_search", "download_paper", "index_papers", "list_papers",
        "library_status", "save_note", "list_notes",
        "search_within_paper", "get_paper_outline", "read_pages",
        "read_chunk_context", "pin_evidence", "get_evidence", "list_evidence",
        "list_project_sources", "list_project_artifacts", "list_project_evidence",
    }
    for t in TOOLS:
        assert t["type"] == "function"
        assert "description" in t["function"]
        assert "parameters" in t["function"]
        assert t["function"]["parameters"]["additionalProperties"] is False


def test_tool_effect_sets_are_derived_from_specs():
    import pragent.tools as tools_module

    assert all(
        spec.effects
        and all(isinstance(effect, ToolEffect) for effect in spec.effects)
        and spec.timeout_seconds > 0
        and isinstance(spec.idempotent, bool)
        for spec in tools_module._REGISTRY.values()
    )
    assert MUTATING_TOOLS == {
        "download_paper", "index_papers", "save_note", "pin_evidence"
    }
    assert EXTERNAL_TOOLS == {"web_search", "download_paper"}
    assert CONFIRMATION_TOOLS == MUTATING_TOOLS | EXTERNAL_TOOLS
    assert tools_module._REGISTRY["save_note"].idempotent is False
    assert tools_module._REGISTRY["download_paper"].timeout_seconds == 180.0
    assert {
        "list_project_sources", "list_project_artifacts", "list_project_evidence"
    }.isdisjoint(CONFIRMATION_TOOLS)
    structured = ToolResult.success(data={"ok": True})
    assert structured.to_model_text() == structured.to_text()


def test_project_read_tools_are_bound_to_session_project(tmp_path):
    database = tmp_path / "project-tools.db"
    store = Store(database)
    repository = ResearchRepository(database)
    project = repository.create_project("Agent 项目")
    source = repository.create_source(
        "url:https://example.org/paper",
        "web",
        title="项目论文",
        authors=["Alice"],
        canonical_url="https://example.org/paper",
        status="ready",
    )
    repository.add_project_source(project.id, source.id)
    artifact = repository.create_artifact(project.id, "review_outline", title="综述提纲")
    ctx = ToolContext(
        store=store,
        embedder=FakeEmbedder(),
        llm=FakeLLM(),
        session_id="session-project",
        project_id=project.id,
        research_repository=repository,
    )

    sources = execute_tool_result("list_project_sources", {"limit": 10}, ctx)
    assert sources.ok and sources.data["items"][0]["source_id"] == source.id
    assert "snapshot_path" not in sources.to_model_text()
    artifacts = execute_tool_result("list_project_artifacts", {"limit": 10}, ctx)
    assert artifacts.ok and artifacts.data["items"][0]["artifact_id"] == artifact.id
    evidence = execute_tool_result("list_project_evidence", {"limit": 10}, ctx)
    assert evidence.ok and evidence.data == []

    unbound = ToolContext(store=store, embedder=FakeEmbedder(), llm=FakeLLM())
    denied = execute_tool_result("list_project_sources", {}, unbound)
    assert denied.code == "project_context_required"


def test_local_search(tmp_path):
    from pragent.tools import ToolContext

    ctx = make_ctx(tmp_path)
    out = execute_tool("local_search", {"query": "注意力机制"}, ctx)
    assert "注意力机制研究" in out
    assert "第1页" in out or "page" in out


def test_local_search_no_hits(tmp_path):
    from pragent.tools import ToolContext

    s = Store(tmp_path / "t.db")  # 空库：混合检索无命中
    ctx = ToolContext(store=s, embedder=FakeEmbedder(), llm=FakeLLM())
    out = execute_tool("local_search", {"query": "任意词"}, ctx)
    assert "未找到" in out


def test_web_search(monkeypatch, tmp_path):
    from pragent import websearch as ws_mod
    from pragent.tools import ToolContext

    monkeypatch.setattr(
        ws_mod,
        "search_papers",
        lambda q, limit, timeout=None: [
            ws_mod.WebPaper(
                title="A Survey", authors=["A"], year=2025, abstract="abs",
                url="http://arxiv.org/abs/2501.1", pdf_url=None,
            )
        ],
    )
    ctx = make_ctx(tmp_path)
    out = execute_tool("web_search", {"query": "llm survey"}, ctx)
    assert "A Survey" in out and "2501.1" in out


def test_web_search_failure(monkeypatch, tmp_path):
    from pragent import websearch as ws_mod
    from pragent.tools import ToolContext

    def boom(q, limit, timeout=None):
        raise ws_mod.WebSearchError("超时")

    monkeypatch.setattr(ws_mod, "search_papers", boom)
    ctx = make_ctx(tmp_path)
    result = execute_tool_result("web_search", {"query": "x"}, ctx)
    assert result.ok is False and result.code == "web_search_failed"
    assert result.retryable is True
    assert "联网检索失败" in result.to_model_text()


def test_unhandled_tool_exception_is_logged_and_sanitized(caplog, tmp_path):
    def handler(ctx, secret=""):
        raise RuntimeError(f"provider rejected {secret}")

    register_tool(
        ToolSpec(
            name="failing_local_tool",
            description="测试未处理异常",
            parameters={
                "type": "object",
                "properties": {"secret": {"type": "string"}},
                "required": ["secret"],
            },
            handler=handler,
            effects=frozenset({ToolEffect.READ_LOCAL}),
            timeout_seconds=5.0,
            idempotent=True,
        )
    )
    ctx = make_ctx(tmp_path)
    ctx.session_id = "session-1"
    try:
        with caplog.at_level(logging.ERROR, logger="pragent.tools"):
            result = execute_tool_result(
                "failing_local_tool",
                {"secret": "private-token"},
                ctx,
                run_id="run-1",
            )
    finally:
        unregister_tool("failing_local_tool")

    assert result.ok is False
    assert result.code == "tool_execution_failed"
    assert result.message == "工具 failing_local_tool 执行失败，请检查本地日志"
    assert "private-token" not in result.to_model_text()
    record = caplog.records[-1]
    assert record.getMessage() == "tool handler failed"
    assert record.session_id == "session-1"
    assert record.run_id == "run-1"
    assert record.tool_name == "failing_local_tool"
    assert record.error_type == "RuntimeError"


def test_web_search_requires_confirmation_before_external_request(monkeypatch, tmp_path):
    from pragent import websearch as ws_mod
    from pragent.tools import confirm_pending_action

    calls = []
    monkeypatch.setattr(ws_mod, "search_papers", lambda q, limit, timeout=None: calls.append(q) or [])
    ctx = make_ctx(tmp_path)
    ctx.require_confirmation = True

    result = execute_tool("web_search", {"query": "private-title", "top": 3}, ctx)

    assert "尚未执行" in result
    assert calls == []
    name, _ = confirm_pending_action(ctx)
    assert name == "web_search"
    assert calls == ["private-title"]


def test_download_paper(monkeypatch, tmp_path):
    from pragent import download as dl_mod
    from pragent.tools import ToolContext

    pdf_bytes = b"%PDF-1.7 fake " * 50

    def fake_download(url, target_dir, timeout=60):
        target = target_dir / "2402.11651.pdf"
        target.write_bytes(pdf_bytes)
        return target

    monkeypatch.setattr(dl_mod, "download_pdf", fake_download)
    indexed = []

    def fake_index_pdf(store, path, embedder, **kwargs):
        indexed.append((path, kwargs))
        return {"added": 1, "updated": 0, "unchanged": 0, "failed": 0}

    monkeypatch.setattr("pragent.indexer.index_pdf", fake_index_pdf)
    monkeypatch.setattr("pragent.config.download_dir_override", lambda: None)
    ctx = make_ctx(tmp_path, library_dir=tmp_path / "lib")
    (tmp_path / "lib").mkdir()
    out = execute_tool("download_paper", {"url": "https://arxiv.org/abs/2402.11651"}, ctx)
    assert "已下载并索引" in out and "2402.11651" in out
    assert (tmp_path / "lib" / "2402.11651.pdf").exists()
    assert indexed[0][0] == tmp_path / "lib" / "2402.11651.pdf"
    assert indexed[0][1]["set_library_dir_if_missing"] is True


def test_download_paper_override_dir_priority(monkeypatch, tmp_path):
    """显式配置目录优先于论文库目录。"""
    from pragent import download as dl_mod
    from pragent.tools import ToolContext

    override = tmp_path / "override"
    pdf_bytes = b"%PDF-1.7 fake " * 50

    def fake_download(url, target_dir, timeout=60):
        target = target_dir / "2402.11651.pdf"
        target.write_bytes(pdf_bytes)
        return target

    monkeypatch.setattr(dl_mod, "download_pdf", fake_download)
    monkeypatch.setattr(
        "pragent.indexer.index_pdf",
        lambda store, path, embedder, **kw: {
            "added": 1, "updated": 0, "unchanged": 0, "failed": 0
        },
    )
    monkeypatch.setattr("pragent.config.download_dir_override", lambda: override)
    lib = tmp_path / "lib"
    lib.mkdir()
    ctx = make_ctx(tmp_path, library_dir=lib)
    out = execute_tool("download_paper", {"url": "https://arxiv.org/abs/2402.11651"}, ctx)
    assert (override / "2402.11651.pdf").exists()
    assert not (lib / "2402.11651.pdf").exists()


def test_download_paper_no_library(monkeypatch, tmp_path):
    monkeypatch.setattr("pragent.config.download_dir_override", lambda: None)
    ctx = make_ctx(tmp_path)  # 无 library_dir
    result = execute_tool_result(
        "download_paper",
        {"url": "https://arxiv.org/abs/2402.11651"},
        ctx,
    )
    assert result.ok is False and result.code == "download_dir_missing"
    assert "未配置下载目录" in result.message and "PRA_DOWNLOAD_DIR" in result.message


def test_index_missing_library_is_a_structured_failure(tmp_path):
    ctx = make_ctx(tmp_path)
    result = execute_tool_result("index_papers", {}, ctx)
    assert result.ok is False
    assert result.code == "library_missing"
    assert "尚未建立论文库" in result.message


def test_download_confirmation_freezes_and_displays_target_directory(monkeypatch, tmp_path):
    from pragent.tools import pending_action_description

    monkeypatch.setattr("pragent.config.download_dir_override", lambda: None)
    library = tmp_path / "lib"
    library.mkdir()
    ctx = make_ctx(tmp_path, library_dir=library)
    ctx.require_confirmation = True

    result = execute_tool(
        "download_paper",
        {"url": "https://arxiv.org/abs/2402.11651"},
        ctx,
    )

    assert str(library.resolve()) not in result
    displayed = json.loads(pending_action_description(ctx).split("：", 1)[1])
    assert displayed["_confirmed_target_dir"] == str(library.resolve())
    assert ctx.pending_action is not None
    assert ctx.pending_action[1]["_confirmed_target_dir"] == str(library.resolve())


def test_list_papers_and_status(tmp_path):
    ctx = make_ctx(tmp_path)
    out = execute_tool("list_papers", {}, ctx)
    assert "共 1 篇" in out and "注意力机制研究" in out
    out = execute_tool("library_status", {}, ctx)
    assert "论文 1 篇" in out


def test_public_tool_results_do_not_expose_absolute_host_paths(monkeypatch, tmp_path):
    notes = tmp_path / "private-notes"
    monkeypatch.setattr("pragent.config.notes_dir", lambda: notes)
    ctx = make_ctx(tmp_path, library_dir=tmp_path / "private-library")
    (tmp_path / "private-library").mkdir()

    search_result = execute_tool("local_search", {"query": "注意力", "top": 1}, ctx)
    papers_result = execute_tool("list_papers", {}, ctx)
    status_result = execute_tool("library_status", {}, ctx)
    note_result = execute_tool(
        "save_note", {"filename": "safe.md", "content": "note"}, ctx
    )

    serialized = "\n".join(
        (search_result, papers_result, status_result, note_result)
    )
    assert str(tmp_path) not in serialized
    assert "a.pdf" in serialized and "safe.md" in serialized


def test_unknown_tool(tmp_path):
    ctx = make_ctx(tmp_path)
    out = execute_tool("no_such_tool", {}, ctx)
    assert "未知工具" in out and "local_search" in out


def test_save_note_creates_dir_and_file(monkeypatch, tmp_path):
    notes = tmp_path / "notes"
    monkeypatch.setattr("pragent.config.notes_dir", lambda: notes)
    ctx = make_ctx(tmp_path)
    out = execute_tool("save_note", {"filename": "总结.md", "content": "这是一篇总结"}, ctx)
    assert "已保存" in out and "总结.md" in out
    assert (notes / "总结.md").read_text(encoding="utf-8") == "这是一篇总结"


def test_save_note_no_overwrite(monkeypatch, tmp_path):
    notes = tmp_path / "notes"
    monkeypatch.setattr("pragent.config.notes_dir", lambda: notes)
    notes.mkdir()
    (notes / "a.md").write_text("v1", encoding="utf-8")
    ctx = make_ctx(tmp_path)
    out = execute_tool("save_note", {"filename": "a.md", "content": "v2"}, ctx)
    assert "a (1).md" in out
    assert (notes / "a.md").read_text(encoding="utf-8") == "v1"
    assert (notes / "a (1).md").read_text(encoding="utf-8") == "v2"


def test_save_note_path_traversal_blocked(monkeypatch, tmp_path):
    notes = tmp_path / "notes"
    monkeypatch.setattr("pragent.config.notes_dir", lambda: notes)
    ctx = make_ctx(tmp_path)
    out = execute_tool("save_note", {"filename": "../../../evil.md", "content": "x"}, ctx)
    assert "已保存" in out
    # 只落在 notes 目录内，且文件名被清洗
    saved = list(notes.glob("*.md"))
    assert len(saved) == 1
    assert saved[0].name == "evil.md"
    assert not (tmp_path.parent / "evil.md").exists()


def test_save_note_invalid_chars(monkeypatch, tmp_path):
    notes = tmp_path / "notes"
    monkeypatch.setattr("pragent.config.notes_dir", lambda: notes)
    ctx = make_ctx(tmp_path)
    out = execute_tool("save_note", {"filename": "a|b?c*.md", "content": "x"}, ctx)
    assert "已保存" in out
    assert (notes / "a_b_c_.md").exists()


def test_list_notes(monkeypatch, tmp_path):
    notes = tmp_path / "notes"
    monkeypatch.setattr("pragent.config.notes_dir", lambda: notes)
    notes.mkdir()
    (notes / "n1.md").write_text("12345", encoding="utf-8")
    ctx = make_ctx(tmp_path)
    out = execute_tool("list_notes", {}, ctx)
    assert "n1.md" in out and "size" in out
    # 空目录
    (notes / "n1.md").unlink()
    assert "空" in execute_tool("list_notes", {}, ctx)
    # 不存在
    monkeypatch.setattr("pragent.config.notes_dir", lambda: tmp_path / "nope")
    assert "不存在" in execute_tool("list_notes", {}, ctx)


def test_tool_error_returns_text(tmp_path):
    ctx = make_ctx(tmp_path)
    out = execute_tool("local_search", {"query": 123}, ctx)  # 参数类型错误
    assert "工具" in out and ("参数" in out or "失败" in out)


def test_mutating_tool_requires_exact_user_confirmation(monkeypatch, tmp_path):
    from pragent.tools import confirm_pending_action

    notes = tmp_path / "notes"
    monkeypatch.setattr("pragent.config.notes_dir", lambda: notes)
    ctx = make_ctx(tmp_path)
    ctx.require_confirmation = True

    out = execute_tool("save_note", {"filename": "safe.md", "content": "v1"}, ctx)
    assert "尚未执行" in out and "/confirm" in out
    assert not (notes / "safe.md").exists()
    assert ctx.pending_action == ("save_note", {"filename": "safe.md", "content": "v1"})

    # 后续模型写操作不能替换用户将要确认的精确参数。
    execute_tool("save_note", {"filename": "changed.md", "content": "evil"}, ctx)
    name, result = confirm_pending_action(ctx)
    assert name == "save_note" and "已保存" in result
    assert (notes / "safe.md").read_text(encoding="utf-8") == "v1"
    assert not (notes / "changed.md").exists()


def test_unclassified_tools_are_rejected_by_registration_and_execution(
    monkeypatch,
    tmp_path,
):
    import pragent.tools as tools_module

    with pytest.raises(ToolValidationError, match="effects 不能为空"):
        ToolSpec(
            name="unclassified",
            description="bad",
            parameters={"type": "object", "properties": {}},
            handler=lambda ctx: "bad",
            effects=frozenset(),
        )
    with pytest.raises(ToolValidationError, match="ToolSpec"):
        register_tool(object())  # type: ignore[arg-type]

    monkeypatch.setitem(tools_module._REGISTRY, "rogue", object())
    result = execute_tool_result("rogue", {}, make_ctx(tmp_path))
    assert result.ok is False
    assert result.code == "tool_unclassified"


@pytest.mark.parametrize(
    "args, expected",
    [
        ({}, "缺少必填字段"),
        ({"query": 123}, "必须是 string"),
        ({"query": "alpha", "top": 0}, "不能小于 1"),
        ({"query": "alpha", "extra": True}, "未知字段"),
    ],
)
def test_tool_arguments_are_strictly_validated(tmp_path, args, expected):
    result = execute_tool_result("local_search", args, make_ctx(tmp_path))
    assert result.ok is False
    assert result.code == "invalid_arguments"
    assert expected in result.message


def test_confirmation_ticket_binds_action_parameters_and_runtime_ids(monkeypatch, tmp_path):
    from pragent.tools import confirm_pending_action

    notes = tmp_path / "notes"
    monkeypatch.setattr("pragent.config.notes_dir", lambda: notes)
    ctx = make_ctx(tmp_path)
    ctx.require_confirmation = True

    pending_result = execute_tool_result(
        "save_note",
        {"filename": "bound.md", "content": "original"},
        ctx,
        tool_call_id="call_7",
        run_id="run_3",
    )
    assert pending_result.requires_confirmation is True
    assert pending_result.action_id and pending_result.digest
    assert ctx.pending_action is not None

    refused = execute_tool_result(
        "save_note",
        {"filename": "changed.md", "content": "evil"},
        ctx,
        confirmed=True,
        action_id="act_wrong",
        digest=pending_result.digest,
    )
    assert refused.code == "confirmation_mismatch"
    assert not notes.exists()

    name, text = confirm_pending_action(ctx)
    assert name == "save_note" and "已保存" in text
    assert (notes / "bound.md").read_text(encoding="utf-8") == "original"
    confirmed = ctx.last_confirmed_action
    assert confirmed is not None
    assert confirmed.action_id == pending_result.action_id
    assert confirmed.digest == pending_result.digest
    assert confirmed.tool_call_id == "call_7"
    assert confirmed.run_id == "run_3"
    assert isinstance(confirmed.result, ToolResult)


def test_confirmation_detects_pending_parameter_tampering(monkeypatch, tmp_path):
    from pragent.tool_protocol import PendingAction
    from pragent.tools import confirm_pending_action

    notes = tmp_path / "notes"
    monkeypatch.setattr("pragent.config.notes_dir", lambda: notes)
    ctx = make_ctx(tmp_path)
    ctx.require_confirmation = True
    execute_tool("save_note", {"filename": "safe.md", "content": "v1"}, ctx)
    assert isinstance(ctx.pending_action, PendingAction)

    ctx.pending_action.args["content"] = "tampered"
    _, result = confirm_pending_action(ctx)
    assert "已发生变化" in result
    assert not notes.exists()
    assert ctx.pending_action is None


def test_local_and_deep_reading_tools_return_stable_evidence_ids(
    monkeypatch,
    tmp_path,
):
    ctx = make_ctx(tmp_path)
    paper = ctx.store.paper_by_path("a.pdf")
    assert paper is not None and paper.id is not None
    chunks = ctx.store.paper_chunks(paper.id)
    first_chunk_id = chunks[0].id
    assert first_chunk_id is not None

    local = execute_tool_result("local_search", {"query": "注意力机制"}, ctx)
    assert local.ok and local.evidence_ids
    assert isinstance(local.data[0]["chunk_id"], int)
    assert local.data[0]["paper_id"] == paper.id
    assert local.data[0]["evidence_id"].startswith("ev_")

    within = execute_tool_result(
        "search_within_paper",
        {"paper_id": paper.id, "query": "Transformer", "top": 2},
        ctx,
    )
    assert within.ok and within.data["paper"]["id"] == paper.id
    assert within.data["hits"][0]["evidence_id"].startswith("ev_")

    outline = execute_tool_result("get_paper_outline", {"paper_id": paper.id}, ctx)
    assert outline.ok and outline.data["pages"]
    assert outline.data["pages"][0]["evidence_ids"][0].startswith("ev_")

    context = execute_tool_result(
        "read_chunk_context",
        {"chunk_id": first_chunk_id, "before": 0, "after": 1},
        ctx,
    )
    assert context.ok and context.data["center_chunk_id"] == first_chunk_id
    assert context.data["chunks"][0]["evidence_id"].startswith("ev_")

    pinned = execute_tool_result(
        "pin_evidence",
        {"chunk_id": first_chunk_id, "annotation": "关键定义"},
        ctx,
    )
    assert pinned.ok and pinned.evidence_ids[0].startswith("ev_")
    evidence_id = pinned.evidence_ids[0]
    assert pinned.data["annotation"] == "关键定义"

    fetched = execute_tool_result(
        "get_evidence", {"evidence_id": evidence_id}, ctx
    )
    listed = execute_tool_result("list_evidence", {"limit": 5}, ctx)
    assert fetched.ok and fetched.data["evidence_id"] == evidence_id
    assert listed.ok and listed.data[0]["evidence_id"] == evidence_id

    monkeypatch.setattr(
        "pragent.pdf.extract_pdf",
        lambda path: (["第一页正文", "第二页正文"], {}),
    )
    monkeypatch.setattr("pragent.tools._sha256_file", lambda path: paper.sha256)
    pages = execute_tool_result(
        "read_pages",
        {"paper_id": paper.id, "start_page": 1, "end_page": 2},
        ctx,
    )
    assert pages.ok
    assert [page["text"] for page in pages.data["pages"]] == ["第一页正文", "第二页正文"]
    assert pages.data["pages"][0]["evidence_ids"]


def test_stale_evidence_is_returned_for_audit_but_not_citable(tmp_path):
    from pragent.models import Chunk

    ctx = make_ctx(tmp_path)
    paper = ctx.store.paper_by_path("a.pdf")
    assert paper is not None and paper.id is not None
    chunk = ctx.store.paper_chunks(paper.id)[0]
    assert chunk.id is not None
    evidence_id = execute_tool_result("pin_evidence", {"chunk_id": chunk.id}, ctx).evidence_ids[0]

    ctx.store.replace_chunks(
        paper.id,
        [Chunk(None, paper.id, 0, 1, "已修改的内容"), Chunk(None, paper.id, 1, 2, "Transformer 使用自注意力。")],
    )

    fetched = execute_tool_result("get_evidence", {"evidence_id": evidence_id}, ctx)
    listed = execute_tool_result("list_evidence", {"limit": 5}, ctx)

    assert fetched.ok and fetched.data["stale"] is True
    assert fetched.evidence_ids == ()
    assert "已过期" in fetched.message
    assert listed.ok and listed.data[0]["stale"] is True
    assert listed.evidence_ids == ()
    assert "已过期" in listed.message


def test_read_pages_rejects_pdf_changed_after_indexing(tmp_path, monkeypatch):
    from pragent.models import Chunk

    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"indexed-version")
    indexed_sha256 = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    ctx = ToolContext(
        store=Store(tmp_path / "changed.db"),
        embedder=FakeEmbedder(),
        llm=FakeLLM(),
        require_confirmation=False,
    )
    paper_id = ctx.store.upsert_paper(
        make_paper(str(pdf_path), sha256=indexed_sha256, title="哈希校验论文")
    )
    ctx.store.replace_chunks(paper_id, [Chunk(None, paper_id, 0, 1, "索引文本")])
    pdf_path.write_bytes(b"changed-version")
    monkeypatch.setattr(
        "pragent.pdf.extract_pdf",
        lambda path: pytest.fail("哈希不匹配时不应读取 PDF"),
    )

    result = execute_tool_result(
        "read_pages", {"paper_id": paper_id, "start_page": 1}, ctx
    )

    assert result.ok is False
    assert result.code == "paper_source_changed"
    assert "重新运行 pra index" in result.message


def test_evidence_id_schema_is_a_stable_string_contract(tmp_path):
    invalid = execute_tool_result("get_evidence", {"evidence_id": "1"}, make_ctx(tmp_path))
    assert invalid.code == "invalid_arguments"
    schema = next(
        tool["function"]["parameters"]
        for tool in TOOLS
        if tool["function"]["name"] == "get_evidence"
    )
    assert schema["properties"]["evidence_id"] == {
        "type": "string",
        "minLength": 3,
        "maxLength": 128,
    }


def test_outline_and_page_reading_have_hard_output_limits(tmp_path):
    class LargePaperStore:
        def paper_by_id(self, paper_id):
            return {
                "id": paper_id,
                "title": "超长论文",
                "authors": [],
                "path": "large.pdf",
                "page_count": 150,
            }

        def paper_chunks(self, paper_id):
            return [
                {
                    "id": page,
                    "paper_id": paper_id,
                    "seq": page - 1,
                    "page": page,
                    "text": "x" * 2_000,
                }
                for page in range(1, 151)
            ]

        def evidence_from_chunk(self, chunk_id):
            return {"id": f"ev_{chunk_id}"}

    ctx = ToolContext(
        store=LargePaperStore(),
        embedder=FakeEmbedder(),
        llm=FakeLLM(),
        require_confirmation=False,
    )
    outline = execute_tool_result(
        "get_paper_outline",
        {"paper_id": 1, "preview_chars": 1_000},
        ctx,
    )
    assert outline.ok
    assert outline.data["truncated"] is True
    assert len(outline.data["pages"]) <= 100
    assert sum(len(page["preview"]) for page in outline.data["pages"]) <= 24_000

    real_ctx = make_ctx(tmp_path)
    too_many_pages = execute_tool_result(
        "read_pages",
        {"paper_id": 1, "start_page": 1, "end_page": 51},
        real_ctx,
    )
    assert too_many_pages.ok is False
    assert too_many_pages.code == "page_range_too_large"


# ---------- Step 26：deadline/cancel 工具执行预算合同 ----------


def _register_probe_tool(name, handler, *, idempotent=True):
    register_tool(
        ToolSpec(
            name=name,
            description="测试预算合同",
            parameters={"type": "object", "properties": {}},
            handler=handler,
            effects=frozenset({ToolEffect.READ_LOCAL}),
            timeout_seconds=2.0,
            idempotent=idempotent,
        )
    )


def test_run_handler_installs_deadline_and_restores_context(tmp_path):
    seen = {}

    def probe(ctx):
        seen["deadline"] = ctx.deadline
        seen["remaining"] = ctx.remaining_seconds()
        seen["runnable"] = ctx.is_runnable()
        return ToolResult.success(message="ok")

    _register_probe_tool("deadline_probe", probe)
    try:
        ctx = make_ctx(tmp_path)
        assert ctx.deadline is None
        result = execute_tool_result("deadline_probe", {}, ctx)
        assert result.ok
        assert seen["deadline"] is not None
        assert 0 < seen["remaining"] <= 2.0
        assert seen["runnable"] is True
        # 执行结束后恢复原状，不污染下一次调用。
        assert ctx.deadline is None
    finally:
        unregister_tool("deadline_probe")


def test_deadline_exceeded_result_follows_idempotency(tmp_path):
    import time as time_mod

    def expired(ctx):
        # 模拟 handler 合作式检查：预算已耗尽时主动停止。
        ctx.deadline = time_mod.monotonic() - 0.01
        ctx.check_deadline()

    _register_probe_tool("expired_idempotent", expired, idempotent=True)
    _register_probe_tool("expired_mutating", expired, idempotent=False)
    try:
        ctx = make_ctx(tmp_path)
        result = execute_tool_result("expired_idempotent", {}, ctx)
        assert result.ok is False
        assert result.code == "tool_deadline_exceeded"
        assert result.retryable is True

        mutating = execute_tool_result("expired_mutating", {}, ctx)
        assert mutating.code == "tool_deadline_exceeded"
        # 非幂等工具超时后副作用未知，不允许自动重试。
        assert mutating.retryable is False
    finally:
        unregister_tool("expired_idempotent")
        unregister_tool("expired_mutating")


def test_cancel_event_stops_cooperative_tool(tmp_path):
    import threading

    def cancellable(ctx):
        ctx.check_cancelled()
        return ToolResult.success(message="不应到达")

    _register_probe_tool("cancellable_probe", cancellable, idempotent=True)
    try:
        ctx = make_ctx(tmp_path)
        result = execute_tool_result("cancellable_probe", {}, ctx)
        assert result.ok is True  # 未取消时正常执行

        ctx.cancel_event = threading.Event()
        ctx.cancel_event.set()
        cancelled = execute_tool_result("cancellable_probe", {}, ctx)
        assert cancelled.ok is False
        assert cancelled.code == "tool_cancelled"
        assert cancelled.retryable is True
    finally:
        unregister_tool("cancellable_probe")


def test_web_search_uses_remaining_budget_as_network_timeout(tmp_path, monkeypatch):
    import time as time_mod

    from pragent import websearch as ws_mod

    captured = {}

    def fake_search(query, limit=5, timeout=None):
        captured["timeout"] = timeout
        return []

    monkeypatch.setattr(ws_mod, "search_papers", fake_search)
    ctx = make_ctx(tmp_path)

    # 默认预算为 web_search 的 spec 超时（30 秒）。
    execute_tool("web_search", {"query": "default budget"}, ctx)
    assert captured["timeout"] is not None
    assert 0 < captured["timeout"] <= 30.0

    # 剩余预算收紧时，网络超时随之收紧。
    ctx.deadline = time_mod.monotonic() + 4.0
    execute_tool("web_search", {"query": "with deadline"}, ctx)
    assert captured["timeout"] is not None
    assert 0 < captured["timeout"] <= 4.0


def test_download_paper_passes_remaining_budget(tmp_path, monkeypatch):
    import time as time_mod

    from pragent import download as dl_mod

    captured = {}

    def fake_download(url, target_dir, timeout=60):
        captured["timeout"] = timeout
        target = target_dir / "2402.11651.pdf"
        target.write_bytes(b"%PDF-1.7 fake " * 50)
        return target

    monkeypatch.setattr(dl_mod, "download_pdf", fake_download)
    monkeypatch.setattr(
        "pragent.indexer.index_pdf",
        lambda store, path, embedder, **kw: {
            "added": 1, "updated": 0, "unchanged": 0, "failed": 0
        },
    )
    monkeypatch.setattr("pragent.config.download_dir_override", lambda: None)
    lib = tmp_path / "lib"
    lib.mkdir()
    ctx = make_ctx(tmp_path, library_dir=lib)
    ctx.deadline = time_mod.monotonic() + 120.0
    execute_tool("download_paper", {"url": "https://arxiv.org/abs/2402.11651"}, ctx)
    # 下载 I/O 同样消费剩余预算而非固定超时。
    assert captured["timeout"] is not None
    assert 0 < captured["timeout"] <= 120.0


def test_index_papers_stops_when_cancelled(tmp_path, monkeypatch):
    import threading

    ctx = make_ctx(tmp_path, library_dir=tmp_path / "lib")
    (tmp_path / "lib").mkdir()
    monkeypatch.setattr(
        "pragent.indexer.index_library",
        lambda store, target, embedder, *, progress=None, should_continue=None: {
            "added": 0,
            "updated": 0,
            "unchanged": 0,
            "failed": 0,
            "removed": 0,
            "skipped_no_text": 0,
            "cancelled": should_continue is not None and not should_continue(),
        },
    )
    ctx.cancel_event = threading.Event()
    ctx.cancel_event.set()
    result = execute_tool_result("index_papers", {}, ctx)
    assert result.ok is False
    assert result.code == "tool_cancelled"
    assert result.retryable is True
    assert result.data["cancelled"] is True
