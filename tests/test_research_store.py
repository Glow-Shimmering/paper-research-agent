import pytest

from pragent.models import Chunk, Paper
from pragent.storage import (
    RecordVersionConflictError,
    ResearchRepository,
    SourceIdentityConflictError,
)
from pragent.store import Store


def make_paper(path="paper.pdf", *, sha256="paper-sha", title="论文"):
    return Paper(
        id=None,
        path=path,
        sha256=sha256,
        title=title,
        authors=["甲", "乙"],
        year=2024,
        page_count=2,
        has_text=True,
        indexed_at="2026-01-01T00:00:00",
    )


def test_project_questions_pagination_cas_and_reopen(tmp_path):
    db_path = tmp_path / "research.db"
    first = ResearchRepository(db_path)
    project = first.create_project("Agent 调研", description="比较研究")
    first.create_project("第二个项目")
    first.create_project("归档项目", status="archived")

    page = first.list_projects(q="项目", limit=1, offset=0)
    assert page.total == 2
    assert len(page.items) == 1
    assert page.limit == 1 and page.offset == 0
    assert first.list_projects(status="archived").total == 1
    with pytest.raises(ValueError, match="1–200"):
        first.list_projects(limit=201)

    q1 = first.create_question(project.id, "核心方法是什么？")
    q2 = first.create_question(project.id, "主要局限是什么？")
    assert [(q.position, q.question) for q in first.list_questions(project.id)] == [
        (0, "核心方法是什么？"),
        (1, "主要局限是什么？"),
    ]
    q1 = first.update_question(
        q1.id,
        expected_version=q1.version,
        question="核心创新是什么？",
        position=2,
    )
    assert q1.version == 2

    second = ResearchRepository(db_path)
    stale = second.get_project(project.id)
    updated = first.update_project(
        project.id, expected_version=project.version, title="Agent 深度调研"
    )
    assert updated.version == 2
    with pytest.raises(RecordVersionConflictError, match="版本冲突"):
        second.update_project(
            project.id, expected_version=stale.version, description="过期写入"
        )

    first.close()
    second.close()
    reopened = ResearchRepository(db_path)
    assert reopened.get_project(project.id).title == "Agent 深度调研"
    assert [q.question for q in reopened.list_questions(project.id)] == [
        "主要局限是什么？",
        "核心创新是什么？",
    ]
    reopened.delete_question(q2.id, expected_version=q2.version)
    assert len(reopened.list_questions(project.id)) == 1
    reopened.close()


def test_sources_provenance_membership_and_local_paper_promotion(tmp_path):
    db_path = tmp_path / "sources.db"
    store = Store(db_path)
    paper_id = store.upsert_paper(
        make_paper(), [Chunk(None, 0, 0, 1, "论文原文")]
    )
    store.close()

    repo = ResearchRepository(db_path)
    project = repo.create_project("来源项目")
    local = repo.ensure_source_for_paper(paper_id)
    same = repo.ensure_source_for_paper(paper_id)
    assert same.id == local.id
    assert local.indexed_paper_id == paper_id
    assert local.status == "ready"
    assert repo.list_source_identities(local.id)[0].normalized_value == "paper-sha"

    web = repo.create_source(
        "url:https://example.org/report",
        "web",
        title="技术报告",
        authors=["Alice"],
        canonical_url="https://example.org/report",
        status="ready",
        metadata={"language": "en"},
    )
    url_identity = repo.add_source_identity(
        web.id, "url", "https://example.org/report", is_primary=True
    )
    assert url_identity.is_primary
    assert repo.add_source_identity(
        web.id, "url", "https://example.org/report"
    ).id == url_identity.id
    with pytest.raises(SourceIdentityConflictError, match="已属于来源"):
        repo.add_source_identity(local.id, "url", "https://example.org/report")

    record = repo.add_source_record(
        web.id,
        "web",
        "https://example.org/report",
        {"title": "旧标题"},
    )
    updated_record = repo.add_source_record(
        web.id,
        "web",
        "https://example.org/report",
        {"title": "技术报告"},
    )
    assert updated_record.id == record.id
    assert updated_record.raw_metadata == {"title": "技术报告"}

    repo.add_project_source(project.id, local.id)
    repo.add_project_source(project.id, web.id)
    memberships = repo.list_project_sources(project.id, limit=1, offset=1)
    assert memberships.total == 2
    assert len(memberships.items) == 1
    assert memberships.items[0].source.id == web.id
    assert repo.list_sources(q="Alice").items == (web,)

    concurrent = ResearchRepository(db_path)
    stale_web = concurrent.get_source(web.id)
    changed_web = repo.update_source(
        web.id,
        expected_version=web.version,
        title="更新后的技术报告",
        snapshot_sha256="snapshot-v2",
    )
    assert changed_web.version == 2
    with pytest.raises(RecordVersionConflictError, match="版本冲突"):
        concurrent.update_source(
            web.id,
            expected_version=stale_web.version,
            title="过期标题",
        )
    concurrent.close()
    repo.close()

    reopened = ResearchRepository(db_path)
    assert reopened.list_project_sources(project.id).total == 2
    assert len(reopened.list_source_records(web.id)) == 1
    reopened.close()


def test_artifact_revisions_evidence_links_and_freshness(tmp_path):
    db_path = tmp_path / "artifacts.db"
    store = Store(db_path)
    paper_id = store.upsert_paper(
        make_paper(), [Chunk(None, 0, 0, 1, "关键原文证据")]
    )
    chunk_id = store.paper_chunks(paper_id)[0].id
    evidence = store.pin_evidence(store.evidence_from_chunk(chunk_id))

    repo = ResearchRepository(db_path)
    project = repo.create_project("精读项目")
    source = repo.ensure_source_for_paper(paper_id)
    repo.add_project_source(project.id, source.id)
    artifact = repo.create_artifact(
        project.id, "deep_read", source_id=source.id, title="精读卡"
    )

    first_revision = repo.append_artifact_revision(
        artifact.id,
        {"research_question": "如何改进检索？"},
        expected_artifact_version=artifact.version,
        created_by="model",
        evidence_links=[(evidence.id, "$.research_question", 0)],
        model="scripted",
        usage={"total_tokens": 10},
        finish_reason="stop",
        prompt_version="deep-read-v1",
        schema_version=1,
    )
    assert first_revision.revision_number == 1
    assert first_revision.parent_revision_id is None
    assert repo.get_current_artifact_revision(artifact.id) == first_revision
    assert repo.list_artifact_evidence(first_revision.id)[0].evidence_id == evidence.id
    assert repo.artifact_freshness(artifact.id).stale is False

    current_artifact = repo.get_artifact(artifact.id)
    with pytest.raises(KeyError, match="Evidence 不存在"):
        repo.append_artifact_revision(
            artifact.id,
            {"bad": True},
            expected_artifact_version=current_artifact.version,
            created_by="user",
            evidence_links=[("ev_missing", "$.bad", 0)],
        )
    assert repo.get_artifact(artifact.id).version == current_artifact.version
    assert repo.list_artifact_revisions(artifact.id).total == 1

    store.upsert_paper(
        make_paper(sha256="changed-sha"),
        [Chunk(None, paper_id, 0, 1, "修改后的原文")],
    )
    stale = repo.artifact_freshness(artifact.id)
    assert stale.stale is True
    assert "来源" in stale.reason

    current_artifact = repo.get_artifact(artifact.id)
    second_revision = repo.append_artifact_revision(
        artifact.id,
        {"research_question": "更新后的问题"},
        expected_artifact_version=current_artifact.version,
        created_by="user",
    )
    assert second_revision.parent_revision_id == first_revision.id
    assert second_revision.revision_number == 2
    assert repo.artifact_freshness(artifact.id).stale is False
    assert [r.revision_number for r in repo.list_artifact_revisions(artifact.id).items] == [
        2,
        1,
    ]

    with pytest.raises(RecordVersionConflictError, match="版本冲突"):
        repo.append_artifact_revision(
            artifact.id,
            {"stale": True},
            expected_artifact_version=current_artifact.version,
            created_by="user",
        )
    repo.close()
    store.close()


def test_research_notes_scope_pagination_and_cas(tmp_path):
    db_path = tmp_path / "notes.db"
    store = Store(db_path)
    paper_id = store.upsert_paper(
        make_paper(), [Chunk(None, 0, 0, 1, "证据文本")]
    )
    evidence = store.pin_evidence(store.evidence_from_chunk(store.paper_chunks(paper_id)[0].id))

    repo = ResearchRepository(db_path)
    project = repo.create_project("笔记项目")
    source = repo.ensure_source_for_paper(paper_id)
    repo.add_project_source(project.id, source.id)
    project_note = repo.create_note(
        project.id, title="项目笔记", content_markdown="# 项目"
    )
    source_note = repo.create_note(
        project.id,
        scope_kind="source",
        source_id=source.id,
        title="来源笔记",
    )
    evidence_note = repo.create_note(
        project.id,
        scope_kind="evidence",
        evidence_id=evidence.id,
        title="证据笔记",
    )
    assert repo.list_notes(project.id).total == 3
    assert repo.list_notes(project.id, source_id=source.id).items == (source_note,)
    assert repo.list_notes(project.id, evidence_id=evidence.id).items == (evidence_note,)

    updated = repo.update_note(
        project_note.id,
        expected_version=project_note.version,
        content_markdown="# 更新",
    )
    assert updated.version == 2 and updated.content_markdown == "# 更新"
    with pytest.raises(RecordVersionConflictError):
        repo.update_note(
            project_note.id,
            expected_version=project_note.version,
            content_markdown="过期",
        )
    with pytest.raises(ValueError, match="scope"):
        repo.create_note(project.id, scope_kind="source")
    repo.close()
    store.close()
