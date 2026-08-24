import re
from pathlib import Path

from fastapi.testclient import TestClient as FastAPITestClient

from pragent.models import Chunk
from pragent.store import Store
from pragent.webapp import create_app

from helpers import FakeEmbedder, make_paper


class OfflineLLM:
    is_configured = False


def TestClient(app, **kwargs):
    kwargs.setdefault("base_url", "http://127.0.0.1")
    return FastAPITestClient(app, **kwargs)


def _make_store(tmp_path):
    store = Store(tmp_path / "workspace.db")
    paper_path = tmp_path / "private-library" / "paper.pdf"
    paper_path.parent.mkdir()
    paper_path.write_bytes(b"not-read-by-this-test")
    paper = make_paper(
        str(paper_path.resolve()),
        title="本地 RAG 论文",
        authors=["Alice", "Bob"],
        year=2024,
    )
    paper.sha256 = "paper-content-sha"
    paper_id = store.upsert_paper(
        paper,
        [Chunk(None, 0, 0, 1, "用于项目选择的正文")],
    )
    return store, paper_id, paper_path


def _app(store, *, api_key=""):
    return create_app(
        store=store,
        embedder=FakeEmbedder(),
        llm=OfflineLLM(),
        api_key=api_key,
    )


def test_project_json_vertical_slice_persists_across_app_restart_and_redacts_paths(
    tmp_path,
):
    store, paper_id, paper_path = _make_store(tmp_path)
    with TestClient(_app(store)) as client:
        created = client.post(
            "/api/v1/projects",
            json={"title": "RAG 评测", "description": "比较评测方法"},
        )
        assert created.status_code == 201
        project = created.json()

        invalid_position = client.post(
            f"/api/v1/projects/{project['id']}/questions",
            json={"question": "无效顺序", "position": 1.5},
        )
        assert invalid_position.status_code == 400

        question = client.post(
            f"/api/v1/projects/{project['id']}/questions",
            json={"question": "主要评测维度是什么？"},
        )
        assert question.status_code == 201
        question = question.json()
        updated = client.patch(
            f"/api/v1/projects/{project['id']}/questions/{question['id']}",
            json={
                "expected_version": question["version"],
                "question": "核心评测维度是什么？",
                "position": 1,
            },
        )
        assert updated.status_code == 200
        assert updated.json()["version"] == 2
        stale = client.patch(
            f"/api/v1/projects/{project['id']}/questions/{question['id']}",
            json={
                "expected_version": question["version"],
                "question": "过期写入",
            },
        )
        assert stale.status_code == 409

        available = client.get(
            f"/api/v1/projects/{project['id']}/available-papers"
        )
        assert available.status_code == 200
        assert available.json()["items"][0]["filename"] == "paper.pdf"
        assert str(paper_path.parent) not in available.text
        assert "path" not in available.json()["items"][0]

        membership = client.post(
            f"/api/v1/projects/{project['id']}/sources",
            json={"paper_id": paper_id},
        )
        assert membership.status_code == 201
        assert membership.json()["source"]["indexed_paper_id"] == paper_id
        assert str(paper_path.parent) not in membership.text

    store.close()
    reopened_store = Store(tmp_path / "workspace.db")
    with TestClient(_app(reopened_store)) as restarted:
        restored = restarted.get(f"/api/v1/projects/{project['id']}")
        assert restored.status_code == 200
        assert restored.json()["title"] == "RAG 评测"
        questions = restarted.get(
            f"/api/v1/projects/{project['id']}/questions"
        ).json()["items"]
        assert questions[0]["question"] == "核心评测维度是什么？"
        sources = restarted.get(
            f"/api/v1/projects/{project['id']}/sources"
        ).json()["items"]
        assert sources[0]["source"]["title"] == "本地 RAG 论文"
        assert str(tmp_path) not in str(sources)
    reopened_store.close()


def test_htmx_workspace_csrf_question_edit_source_selection_and_refresh(tmp_path):
    store, _, paper_path = _make_store(tmp_path)
    with TestClient(_app(store), follow_redirects=False) as client:
        page = client.get("/ui/projects")
        assert page.status_code == 200
        assert "研究项目" in page.text
        assert "/ui/static/htmx.min.js" in page.text
        assert "HttpOnly" in page.headers["set-cookie"]
        assert "SameSite=strict" in page.headers["set-cookie"]
        csrf = client.cookies["pra_csrf"]

        rejected = client.post("/ui/projects", data={"title": "无 token"})
        assert rejected.status_code == 403
        cross_origin = client.post(
            "/ui/projects",
            data={"csrf_token": csrf, "title": "跨站"},
            headers={"Origin": "https://evil.example"},
        )
        assert cross_origin.status_code == 403

        created = client.post(
            "/ui/projects",
            data={
                "csrf_token": csrf,
                "title": "HTMX 项目",
                "description": "刷新后仍存在",
            },
        )
        assert created.status_code == 303
        workspace_url = created.headers["location"]
        project_id = workspace_url.rsplit("/", 1)[-1]

        workspace = client.get(workspace_url)
        assert workspace.status_code == 200
        assert "HTMX 项目" in workspace.text
        assert "本地 RAG 论文" in workspace.text
        assert str(paper_path.parent) not in workspace.text

        question_fragment = client.post(
            f"/ui/projects/{project_id}/questions",
            data={"csrf_token": csrf, "question": "如何评测？"},
            headers={"HX-Request": "true"},
        )
        assert question_fragment.status_code == 200
        assert 'id="question-list"' in question_fragment.text
        assert "如何评测？" in question_fragment.text
        question_id = re.search(
            rf"/ui/projects/{re.escape(project_id)}/questions/([^\"/]+)\"",
            question_fragment.text,
        ).group(1)
        version = int(
            re.search(
                r'name="expected_version" value="(\d+)"',
                question_fragment.text,
            ).group(1)
        )

        updated_fragment = client.post(
            f"/ui/projects/{project_id}/questions/{question_id}",
            data={
                "csrf_token": csrf,
                "expected_version": version,
                "position": 2,
                "question": "如何可靠评测？",
            },
            headers={"HX-Request": "true"},
        )
        assert updated_fragment.status_code == 200
        assert "如何可靠评测？" in updated_fragment.text

        source_fragment = client.post(
            f"/ui/projects/{project_id}/sources",
            data={"csrf_token": csrf, "paper_id": 1},
            headers={"HX-Request": "true"},
        )
        assert source_fragment.status_code == 200
        assert 'id="source-panels"' in source_fragment.text
        assert "已选来源（1）" in source_fragment.text
        assert str(tmp_path) not in source_fragment.text

        refreshed = client.get(workspace_url)
        assert "如何可靠评测？" in refreshed.text
        assert "已选来源（1）" in refreshed.text
    store.close()


def test_question_cannot_be_mutated_through_another_project(tmp_path):
    store, _, _ = _make_store(tmp_path)
    with TestClient(_app(store)) as client:
        first = client.post("/api/v1/projects", json={"title": "项目一"}).json()
        second = client.post("/api/v1/projects", json={"title": "项目二"}).json()
        question = client.post(
            f"/api/v1/projects/{second['id']}/questions",
            json={"question": "项目二的问题"},
        ).json()

        response = client.patch(
            f"/api/v1/projects/{first['id']}/questions/{question['id']}",
            json={
                "expected_version": question["version"],
                "question": "越权修改",
            },
        )
        assert response.status_code == 404
        unchanged = client.get(
            f"/api/v1/projects/{second['id']}/questions"
        ).json()["items"][0]
        assert unchanged["question"] == "项目二的问题"
    store.close()


def test_research_ui_remote_access_is_fail_closed_but_static_asset_is_public(tmp_path):
    store, _, _ = _make_store(tmp_path)
    with TestClient(_app(store, api_key="secret")) as client:
        assert client.get("/ui/projects").status_code == 401
        assert client.get(
            "/ui/projects", headers={"X-PRA-Key": "secret"}
        ).status_code == 200
        authenticated = client.post(
            "/api/ui-auth", headers={"X-PRA-Key": "secret"}
        )
        assert authenticated.status_code == 200
        assert "HttpOnly" in authenticated.headers["set-cookie"]
        assert "secret" not in authenticated.headers["set-cookie"]
        assert client.get("/ui/projects").status_code == 200
        asset = client.get("/ui/static/htmx.min.js")
        assert asset.status_code == 200
        media_type = asset.headers["content-type"].split(";", 1)[0]
        assert media_type in {"text/javascript", "application/javascript"}
    store.close()
