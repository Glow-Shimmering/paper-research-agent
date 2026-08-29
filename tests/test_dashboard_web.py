"""Dashboard、任务中心、证据与笔记的 JSON/HTMX UI 合同（Step 27）。"""

from fastapi.testclient import TestClient as FastAPITestClient

from pragent.jobs import JobQueue
from pragent.models import Chunk
from pragent.storage import JobRepository, ResearchRepository
from pragent.store import Store
from pragent.webapp import create_app

from helpers import FakeEmbedder, make_paper


class OfflineLLM:
    is_configured = False


def TestClient(app, **kwargs):
    kwargs.setdefault("base_url", "http://127.0.0.1")
    return FastAPITestClient(app, **kwargs)


def _make_app(tmp_path):
    store = Store(tmp_path / "dash.db")
    paper_path = tmp_path / "private-library" / "paper.pdf"
    paper_path.parent.mkdir()
    paper_path.write_bytes(b"not-read-by-tests")
    paper = make_paper(
        str(paper_path.resolve()),
        title="本地 RAG 论文",
        authors=["Alice"],
        year=2024,
    )
    paper.sha256 = "paper-content-sha"
    paper_id = store.upsert_paper(
        paper, [Chunk(None, 0, 0, 1, "用于证据与笔记合同的正文片段")]
    )
    app = create_app(store=store, embedder=FakeEmbedder(), llm=OfflineLLM())
    return app, store, paper_id, paper_path


def _enqueue_job(tmp_path, project_id):
    queue = JobQueue(JobRepository(tmp_path / "dash.db"))
    return queue.enqueue(
        "deep_read", {"project_id": project_id}, project_id=project_id
    )


def test_dashboard_help_and_empty_states(tmp_path):
    app, store, _, _ = _make_app(tmp_path)
    client = TestClient(app)

    dashboard = client.get("/ui/")
    assert dashboard.status_code == 200
    assert "还没有研究项目" in dashboard.text
    assert "任务中心" in dashboard.text and "帮助" in dashboard.text

    jobs = client.get("/ui/jobs")
    assert jobs.status_code == 200
    assert "没有符合条件的任务" in jobs.text

    fragment = client.get("/ui/jobs/fragment")
    assert fragment.status_code == 200
    assert "没有符合条件的任务" in fragment.text

    help_page = client.get("/ui/help")
    assert help_page.status_code == 200
    assert "单篇精读" in help_page.text and "多篇比较与综述" in help_page.text
    # 隐私边界必须明示：云端模型会收到问题与选中片段。
    assert "命中" in help_page.text or "选中" in help_page.text

    error_page = client.get("/ui/projects/does-not-exist/evidence")
    assert error_page.status_code == 404
    assert "任务中心" in error_page.text
    store.close()


def test_dashboard_lists_project_source_and_active_job(tmp_path):
    app, store, paper_id, _ = _make_app(tmp_path)
    client = TestClient(app)
    repository = ResearchRepository(tmp_path / "dash.db")
    project = repository.create_project("RAG 评测")
    repository.add_paper_to_project(project.id, paper_id)
    job = _enqueue_job(tmp_path, project.id)

    dashboard = client.get("/ui/")
    assert dashboard.status_code == 200
    assert "RAG 评测" in dashboard.text
    assert "本地 RAG 论文" in dashboard.text
    assert "单篇精读" in dashboard.text and "queued" in dashboard.text

    store.close()


def test_jobs_json_list_and_cancellation_flow(tmp_path):
    app, store, _, _ = _make_app(tmp_path)
    client = TestClient(app)
    repository = ResearchRepository(tmp_path / "dash.db")
    project = repository.create_project("任务项目")
    job = _enqueue_job(tmp_path, project.id)

    listed = client.get("/api/v1/jobs", params={"status": "queued"})
    assert listed.status_code == 200
    items = listed.json()["items"]
    assert items and items[0]["id"] == job.id
    assert items[0]["type_label"] == "单篇精读"

    invalid_status = client.get("/api/v1/jobs", params={"status": "bogus"})
    assert invalid_status.status_code == 400

    single = client.get(f"/api/v1/jobs/{job.id}")
    assert single.status_code == 200
    assert single.json()["status"] == "queued"

    cancelled = client.post(
        f"/api/v1/jobs/{job.id}/cancellation",
        json={"expected_version": job.version},
    )
    assert cancelled.status_code == 202
    assert cancelled.json()["status"] == "cancelled"

    conflict = client.post(
        f"/api/v1/jobs/{job.id}/cancellation",
        json={"expected_version": job.version},
    )
    assert conflict.status_code == 409

    missing = client.post(
        "/api/v1/jobs/job_missing/cancellation",
        json={"expected_version": 1},
    )
    assert missing.status_code == 404
    store.close()


def test_jobs_ui_cancel_with_csrf_and_polling_fragment(tmp_path):
    app, store, _, _ = _make_app(tmp_path)
    client = TestClient(app)
    repository = ResearchRepository(tmp_path / "dash.db")
    project = repository.create_project("取消项目")
    job = _enqueue_job(tmp_path, project.id)
    other = _enqueue_job(tmp_path, project.id)

    page = client.get("/ui/jobs")
    assert page.status_code == 200
    assert "请求取消" in page.text
    csrf = client.cookies["pra_csrf"]

    cancelled = client.post(
        f"/ui/jobs/{job.id}/cancel",
        data={"csrf_token": csrf, "expected_version": str(job.version)},
        headers={"HX-Request": "true"},
    )
    assert cancelled.status_code == 200
    assert "cancelled" in cancelled.text

    # 非 HTMX 请求重定向回任务中心。
    without_htmx = client.post(
        f"/ui/jobs/{other.id}/cancel",
        data={"csrf_token": csrf, "expected_version": str(other.version)},
        follow_redirects=False,
    )
    assert without_htmx.status_code == 303

    conflict = client.post(
        f"/ui/jobs/{job.id}/cancel",
        data={"csrf_token": csrf, "expected_version": str(job.version)},
        headers={"HX-Request": "true"},
    )
    assert conflict.status_code == 409
    store.close()


def test_notes_json_create_update_and_cas_conflict(tmp_path):
    app, store, paper_id, paper_path = _make_app(tmp_path)
    client = TestClient(app)
    repository = ResearchRepository(tmp_path / "dash.db")
    project = repository.create_project("笔记项目")
    membership = repository.add_paper_to_project(project.id, paper_id)

    created = client.post(
        f"/api/v1/projects/{project.id}/notes",
        json={"title": "评测质疑", "content_markdown": "注意基线选择"},
    )
    assert created.status_code == 201
    note = created.json()
    assert note["scope_kind"] == "project"

    source_note = client.post(
        f"/api/v1/projects/{project.id}/notes",
        json={
            "scope_kind": "source",
            "source_id": membership.source.id,
            "title": "来源级笔记",
        },
    )
    assert source_note.status_code == 201

    bad_source = client.post(
        f"/api/v1/projects/{project.id}/notes",
        json={"scope_kind": "source", "source_id": "source-unknown"},
    )
    assert bad_source.status_code == 400

    updated = client.patch(
        f"/api/v1/projects/{project.id}/notes/{note['id']}",
        json={"expected_version": note["version"], "title": "评测质疑（更新）"},
    )
    assert updated.status_code == 200
    assert updated.json()["version"] == 2

    stale = client.patch(
        f"/api/v1/projects/{project.id}/notes/{note['id']}",
        json={"expected_version": note["version"], "title": "过期写入"},
    )
    assert stale.status_code == 409

    listed = client.get(f"/api/v1/projects/{project.id}/notes")
    assert listed.status_code == 200
    assert listed.json()["total"] == 2
    assert str(paper_path.parent) not in listed.text
    store.close()


def test_notes_ui_create_and_edit_with_csrf(tmp_path):
    app, store, paper_id, _ = _make_app(tmp_path)
    client = TestClient(app)
    repository = ResearchRepository(tmp_path / "dash.db")
    project = repository.create_project("笔记 UI 项目")
    repository.add_paper_to_project(project.id, paper_id)

    page = client.get(f"/ui/projects/{project.id}/evidence")
    assert page.status_code == 200
    assert "研究笔记" in page.text and "还没有被产物引用的证据" in page.text
    csrf = client.cookies["pra_csrf"]

    created = client.post(
        f"/ui/projects/{project.id}/notes",
        data={
            "csrf_token": csrf,
            "title": "阅读心得",
            "content_markdown": "先核对数据集口径",
        },
        headers={"HX-Request": "true"},
    )
    assert created.status_code == 200
    assert "阅读心得" in created.text and "v1" in created.text

    notes = repository.list_notes(project.id, limit=10).items
    note = next(item for item in notes if item.title == "阅读心得")

    updated = client.post(
        f"/ui/projects/{project.id}/notes/{note.id}",
        data={
            "csrf_token": csrf,
            "expected_version": str(note.version),
            "title": "阅读心得",
            "content_markdown": "补充：核对检索命中率",
        },
        headers={"HX-Request": "true"},
    )
    assert updated.status_code == 200
    assert "补充：核对检索命中率" in updated.text

    stale = client.post(
        f"/ui/projects/{project.id}/notes/{note.id}",
        data={
            "csrf_token": csrf,
            "expected_version": str(note.version),
            "title": "过期写入",
            "content_markdown": "",
        },
        headers={"HX-Request": "true"},
    )
    assert stale.status_code == 409

    missing_csrf = client.post(
        f"/ui/projects/{project.id}/notes",
        data={"title": "跨站"},
        headers={"HX-Request": "true"},
    )
    assert missing_csrf.status_code == 403
    store.close()


def test_evidence_page_and_json_list_linked_evidence_without_host_paths(tmp_path):
    app, store, paper_id, paper_path = _make_app(tmp_path)
    client = TestClient(app)
    repository = ResearchRepository(tmp_path / "dash.db")
    project = repository.create_project("证据项目")
    repository.add_paper_to_project(project.id, paper_id)

    chunks = store.paper_chunks(paper_id)
    evidence = store.pin_evidence(chunks[0].id)
    artifact = repository.create_artifact(project.id, "deep_read", title="精读卡")
    repository.append_artifact_revision(
        artifact.id,
        {"research_question": {"text": "RAG 评测要点"}},
        expected_artifact_version=artifact.version,
        created_by="user",
        evidence_links=[(evidence.evidence_id, "research_question", 1)],
    )

    page = client.get(f"/ui/projects/{project.id}/evidence")
    assert page.status_code == 200
    assert evidence.evidence_id in page.text
    assert "用于证据与笔记合同的正文片段" in page.text
    assert "paper.pdf" in page.text
    assert str(paper_path.parent) not in page.text

    payload = client.get(f"/api/v1/projects/{project.id}/evidence")
    assert payload.status_code == 200
    items = payload.json()["items"]
    assert items and items[0]["evidence_id"] == evidence.evidence_id
    assert items[0]["locator"] == "paper.pdf"
    assert items[0]["field_path"] == "research_question"
    assert str(paper_path.parent) not in payload.text

    unknown = client.get("/api/v1/projects/missing/evidence")
    assert unknown.status_code == 404
    store.close()
