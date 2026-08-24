import json
import time

from fastapi.testclient import TestClient as FastAPITestClient

from helpers import FakeEmbedder, make_paper
from pragent.models import Chunk
from pragent.storage import JobRepository, ResearchRepository
from pragent.store import Store
from pragent.webapp import create_app


class WebDeepReadLLM:
    is_configured = True
    model = "scripted-web-deep-read"

    def __init__(self):
        self.calls = 0

    def chat_with_metadata(self, system, user):
        self.calls += 1
        if "精读助手" in system:
            payload = json.loads(user)
            evidence = payload["evidence"]
            field = payload["field"]
            content = {
                "text": f"{field} summary call {self.calls}",
                "evidence_refs": [
                    {
                        "evidence_id": evidence[0]["evidence_id"],
                        "quote": "Exact web evidence.",
                    }
                ],
                "insufficient_evidence": False,
            }
        else:
            content = json.loads(user)
        return {
            "content": json.dumps(content, ensure_ascii=False),
            "metadata": {
                "usage": {"total_tokens": 5},
                "finish_reason": "stop",
                "response_id": f"web-{self.calls}",
            },
        }


def TestClient(app, **kwargs):
    kwargs.setdefault("base_url", "http://127.0.0.1")
    return FastAPITestClient(app, **kwargs)


def _environment(tmp_path):
    db_path = tmp_path / "artifact-web.db"
    store = Store(db_path)
    paper_id = store.upsert_paper(
        make_paper(str(tmp_path / "private" / "paper.pdf"), title="Web Deep Paper"),
    )
    text = "Exact web evidence. The method reports results and limitations."
    store.replace_chunks(
        paper_id,
        [Chunk(None, paper_id, 0, 3, text, FakeEmbedder.vecs_for(text))],
    )
    repository = ResearchRepository(db_path)
    project = repository.create_project("精读 Web 项目")
    source = repository.ensure_source_for_paper(paper_id)
    repository.add_project_source(project.id, source.id)
    jobs = JobRepository(db_path)
    app = create_app(
        store=store,
        research_repository=repository,
        job_repository=jobs,
        embedder=FakeEmbedder(),
        llm=WebDeepReadLLM(),
        job_worker_count=1,
    )
    return store, repository, jobs, project, source, app


def _wait_job(client, project_id, job_id):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        payload = client.get(f"/api/v1/projects/{project_id}/jobs/{job_id}").json()
        if payload["terminal"]:
            return payload
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def test_deep_read_api_job_progress_nine_fields_edit_regenerate_and_history(tmp_path):
    store, repository, jobs, project, source, app = _environment(tmp_path)
    with TestClient(app) as client:
        generated = client.post(
            f"/api/v1/projects/{project.id}/sources/{source.id}/deep-reads"
        )
        assert generated.status_code == 202
        artifact = generated.json()["artifact"]
        job = generated.json()["job"]
        terminal = _wait_job(client, project.id, job["id"])
        assert terminal["status"] == "succeeded"
        assert "payload" not in terminal and "lease_owner" not in terminal

        detail = client.get(
            f"/api/v1/projects/{project.id}/deep-reads/{artifact['id']}"
        )
        assert detail.status_code == 200
        payload = detail.json()
        assert [item["name"] for item in payload["fields"]] == [
            "research_question",
            "related_work",
            "core_method",
            "contributions",
            "datasets_and_experiments",
            "main_results",
            "limitations",
            "future_work",
            "key_evidence",
        ]
        before = {item["name"]: item["text"] for item in payload["fields"]}
        version = payload["artifact"]["version"]
        edited = client.patch(
            f"/api/v1/projects/{project.id}/deep-reads/{artifact['id']}/fields/limitations",
            json={"expected_artifact_version": version, "text": "人工补充局限性"},
        )
        assert edited.status_code == 200
        assert edited.json()["fields_by_name"]["limitations"]["text"] == "人工补充局限性"
        stale_edit = client.patch(
            f"/api/v1/projects/{project.id}/deep-reads/{artifact['id']}/fields/limitations",
            json={"expected_artifact_version": version, "text": "过期编辑"},
        )
        assert stale_edit.status_code == 409

        current = edited.json()
        regeneration = client.post(
            f"/api/v1/projects/{project.id}/deep-reads/{artifact['id']}/fields/core_method/regenerations"
        )
        assert regeneration.status_code == 202
        terminal = _wait_job(client, project.id, regeneration.json()["job"]["id"])
        assert terminal["status"] == "succeeded"
        regenerated = client.get(
            f"/api/v1/projects/{project.id}/deep-reads/{artifact['id']}"
        ).json()
        after = {item["name"]: item["text"] for item in regenerated["fields"]}
        assert after["core_method"] != current["fields_by_name"]["core_method"]["text"]
        for name in before:
            if name not in {"core_method", "limitations"}:
                assert after[name] == before[name]
        assert after["limitations"] == "人工补充局限性"
        history = client.get(
            f"/api/v1/projects/{project.id}/deep-reads/{artifact['id']}/revisions"
        ).json()["items"]
        assert [item["revision_number"] for item in history] == [3, 2, 1]

        revision_id = regenerated["revision_public"]["id"]
        evidence = client.get(
            f"/api/v1/projects/{project.id}/deep-reads/{artifact['id']}/revisions/{revision_id}/fields/core_method/evidence"
        )
        assert evidence.status_code == 200
        serialized = evidence.text
        assert "Exact web evidence." in serialized
        assert str(tmp_path) not in serialized
        assert "paper_id" not in serialized and "path" not in serialized

        paper = store.paper_by_id(source.indexed_paper_id)
        changed = make_paper(paper.path, title=paper.title)
        changed.sha256 = "changed-source-sha"
        store.upsert_paper(
            changed,
            [Chunk(None, paper.id, 0, 3, "Changed source text.", FakeEmbedder.vecs_for("Changed source text."))],
        )
        stale = client.get(
            f"/api/v1/projects/{project.id}/deep-reads/{artifact['id']}"
        ).json()
        assert stale["freshness"]["stale"] is True
        historical = client.get(
            f"/api/v1/projects/{project.id}/deep-reads/{artifact['id']}/revisions/{revision_id}/fields/core_method/evidence"
        )
        assert historical.status_code == 200
        assert historical.json()["items"][0]["stale"] is True
    jobs.close()
    repository.close()
    store.close()


def test_deep_read_ui_is_chinese_csrf_protected_and_scope_safe(tmp_path):
    store, repository, jobs, project, source, app = _environment(tmp_path)
    other = repository.create_project("其他项目")
    with TestClient(app) as client:
        listing = client.get(f"/ui/projects/{project.id}/deep-reads")
        assert listing.status_code == 200
        assert "单篇精读" in listing.text and "生成精读卡" in listing.text
        missing_csrf = client.post(
            f"/ui/projects/{project.id}/sources/{source.id}/deep-reads", data={"x": "1"}
        )
        assert missing_csrf.status_code == 403
        generated = client.post(
            f"/api/v1/projects/{project.id}/sources/{source.id}/deep-reads"
        ).json()
        artifact_id = generated["artifact"]["id"]
        _wait_job(client, project.id, generated["job"]["id"])
        page = client.get(f"/ui/projects/{project.id}/deep-reads/{artifact_id}")
        assert page.status_code == 200
        for label in (
            "研究问题", "相关工作", "核心方法", "创新点", "数据集与实验",
            "主要结果", "局限性", "未来工作", "关键原文证据",
        ):
            assert label in page.text
        forged = client.get(
            f"/api/v1/projects/{other.id}/deep-reads/{artifact_id}"
        )
        assert forged.status_code == 404
    jobs.close()
    repository.close()
    store.close()
