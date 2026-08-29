"""Step 28 确定性产品场景：四条核心用户旅程的离线端到端回归。

对应 docs/plans/product-roadmap.md「Manual product journeys」：
1. 单篇精读：建项目→导入 PDF→研究问题→九栏精读卡→证据抽屉→人工编辑→导出→重启恢复；
2. 多篇比较/综述：3 篇论文→精读卡→比较矩阵→综述提纲→章节草稿→切换引用样式→导出；
3. 发现与入库：多 provider 检索→dedupe→下载 PDF→抓取网页→同一次 hybrid search→加入项目；
4. 新鲜度与恢复：重新索引→stale→重生成；任务重启 interrupted→幂等重排；Agent 待确认重启后可继续。

全部使用 FakeEmbedder、脚本化 LLM、fixture provider/fetcher/downloader，无网络。
这些场景回归的是产品合同（schema、预算、证据范围、恢复边界），
不代表真实模型质量，也不能替代需要用户授权的 live DeepSeek/provider smoke。
"""

import csv
import io
import json
import time
import zipfile

import pytest
from fastapi.testclient import TestClient as FastAPITestClient

from helpers import FakeEmbedder, make_pdf, noop_progress
from pragent.exporting import ArtifactExportService, ExportEnvelope
from pragent.indexer import index_library
from pragent.ingestion.safe_fetch import SafeFetchResult
from pragent.ingestion.snapshots import SnapshotStore
from pragent.jobs import JobQueue, WorkerPool
from pragent.research import (
    DEEP_READ_FIELD_ORDER,
    ComparisonArtifactService,
    ComparisonWorkflow,
    DeepReadArtifactService,
    DeepReadCard,
    DeepReadWorkflow,
    ReviewOutlineArtifactService,
    ReviewOutlineWorkflow,
    ReviewSectionArtifactService,
    ReviewSectionWorkflow,
)
from pragent.search import hybrid_search
from pragent.sources import NormalizedSource
from pragent.storage import JobRepository, ResearchRepository
from pragent.store import Store
from pragent.tool_protocol import ToolEffect, ToolResult, ToolSpec
from pragent.tools import register_tool, unregister_tool
from pragent.webapp import create_app


def TestClient(app, **kwargs):
    kwargs.setdefault("base_url", "http://127.0.0.1")
    return FastAPITestClient(app, **kwargs)


class OfflineLLM:
    is_configured = False


class JourneyLLM:
    """脚本化 LLM：按 system 标记扮演精读字段/精读 reduce/综述提纲/综述章节。"""

    model = "scripted-journey-llm"
    is_configured = True

    def __init__(self):
        self.calls = []
        self._last_valid = None

    def chat_with_metadata(self, system, user):
        self.calls.append((system, user))
        if "修复下列 JSON" in system:
            content = json.dumps(self._last_valid, ensure_ascii=False)
        elif "精读助手" in system:
            payload = json.loads(user)
            evidence = payload["evidence"]
            if evidence:
                self._last_valid = {
                    "text": f"{payload['label']}：基于原文的中文总结。",
                    "evidence_refs": [
                        {
                            "evidence_id": evidence[0]["evidence_id"],
                            "quote": evidence[0]["text"][:80],
                        }
                    ],
                    "insufficient_evidence": False,
                }
            else:
                self._last_valid = {
                    "text": "证据不足",
                    "evidence_refs": [],
                    "insufficient_evidence": True,
                }
        elif "综述写作助手" in system:
            section = json.loads(user)["section"]
            tokens = [
                dict(item) for item in section["planned_claims"][0]["evidence_refs"]
            ]
            self._last_valid = {
                "claims": [
                    {
                        "key": "method_synthesis",
                        "text": "所选论文在方法设计上可以互相印证。",
                        "citation_tokens": tokens,
                        "insufficient_evidence": False,
                    }
                ]
            }
        elif '"comparison"' in user:
            payload = json.loads(user)
            comparison = payload["comparison"]
            source_ids = comparison["source_ids"]
            refs = []
            for source_id in source_ids:
                cell = next(
                    item
                    for item in comparison["cells"]
                    if item["source_id"] == source_id and item["evidence_refs"]
                )
                refs.append({"source_id": source_id, **cell["evidence_refs"][0]})
            claim = {
                "text": "所选论文在核心方法与主要结果上互相支持。",
                "source_ids": source_ids,
                "evidence_refs": refs,
                "insufficient_evidence": False,
            }
            self._last_valid = {
                "title": "证据约束的综述提纲",
                "sections": [
                    {
                        "key": "methods",
                        "title": "方法比较",
                        "objective": "比较核心方法",
                        "source_ids": source_ids,
                        "planned_claims": [claim],
                    },
                    {
                        "key": "results",
                        "title": "结果与局限",
                        "objective": "比较主要结果",
                        "source_ids": source_ids,
                        "planned_claims": [claim],
                    },
                ],
            }
        else:
            # Deep Read reduce：回传完整九栏 JSON。
            self._last_valid = json.loads(user)
        return {
            "content": json.dumps(self._last_valid, ensure_ascii=False),
            "metadata": {
                "usage": {"total_tokens": 5},
                "finish_reason": "stop",
                "response_id": f"journey-{len(self.calls)}",
            },
        }


def _wait_terminal_job(client, project_id, job_id, *, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = client.get(f"/api/v1/projects/{project_id}/jobs/{job_id}").json()
        if job["terminal"]:
            return job
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} did not finish in time")


def _index_paper(tmp_path, name, pages, title):
    papers = tmp_path / "papers"
    papers.mkdir(exist_ok=True)
    make_pdf(papers / name, pages, {"title": title, "author": "Alice"})
    return papers


# ---------------------------------------------------------------------------
# 旅程 1：单篇精读
# ---------------------------------------------------------------------------


def test_journey_single_paper_deep_read_edit_export_and_restart(tmp_path):
    papers = _index_paper(
        tmp_path,
        "journey-deep.pdf",
        [
            "Research question: how do evidence grounded agents cite sources?",
            "Related work surveys retrieval augmented generation agents.",
            "Method: field specific retrieval with structured deep read cards.",
            "Innovation: every claim keeps a verifiable evidence identifier.",
            "Experiments run on Dataset A with citation accuracy metrics.",
            "Results: citation accuracy improves on Dataset A benchmarks.",
            "Limitations: text only extraction without tables or figures.",
            "Future work studies multimodal evidence and table understanding.",
        ],
        "Journey Deep Read Paper",
    )
    db_path = tmp_path / "journey1.db"
    store = Store(db_path)
    indexed = index_library(store, papers, FakeEmbedder(), progress=noop_progress)
    assert indexed["added"] == 1
    _, papers_list = store.list_papers(None, 10, 0)
    paper_id = papers_list[0].id

    repository = ResearchRepository(db_path)
    app = create_app(
        store=store,
        research_repository=repository,
        embedder=FakeEmbedder(),
        llm=JourneyLLM(),
    )
    with TestClient(app) as client:
        project = client.post(
            "/api/v1/projects", json={"title": "单篇精读旅程"}
        ).json()
        project_id = project["id"]
        question = client.post(
            f"/api/v1/projects/{project_id}/questions",
            json={"question": "证据如何支撑结论？"},
        )
        assert question.status_code == 201

        membership = client.post(
            f"/api/v1/projects/{project_id}/sources", json={"paper_id": paper_id}
        )
        assert membership.status_code == 201
        source_id = membership.json()["source"]["id"]

        generated = client.post(
            f"/api/v1/projects/{project_id}/sources/{source_id}/deep-reads"
        )
        assert generated.status_code == 202
        artifact_id = generated.json()["artifact"]["id"]
        version = generated.json()["artifact"]["version"]

        job = _wait_terminal_job(client, project_id, generated.json()["job"]["id"])
        assert job["status"] == "succeeded"

        detail = client.get(
            f"/api/v1/projects/{project_id}/deep-reads/{artifact_id}"
        ).json()
        version = detail["artifact"]["version"]
        assert [field["name"] for field in detail["fields"]] == list(
            DEEP_READ_FIELD_ORDER
        )
        assert detail["freshness"]["stale"] is False
        assert detail["revision_public"]["created_by"] == "model"
        assert detail["revision_public"]["model"] == "scripted-journey-llm"
        evidence_bearing = [
            field for field in detail["fields"] if field["evidence_count"] > 0
        ]
        assert evidence_bearing, "旅程要求至少一个栏目携带证据"

        first_field = evidence_bearing[0]["name"]
        drawer = client.get(
            f"/api/v1/projects/{project_id}/deep-reads/{artifact_id}"
            f"/revisions/{detail['revision_public']['id']}"
            f"/fields/{first_field}/evidence"
        ).json()
        assert drawer["items"]
        item = drawer["items"][0]
        assert item["evidence_id"] and item["page"] >= 1 and item["quote"]

        edited = client.patch(
            f"/api/v1/projects/{project_id}/deep-reads/{artifact_id}"
            f"/fields/{first_field}",
            json={
                "expected_artifact_version": version,
                "text": "人工复核后的栏目结论。",
            },
        )
        assert edited.status_code == 200

        exports = tmp_path / "journey1-exports"
        service = ArtifactExportService(repository, store)
        markdown_files = service.export_current(artifact_id, "markdown", exports)
        docx_files = service.export_current(artifact_id, "docx", exports)
        assert markdown_files and docx_files
        markdown_text = markdown_files[0].path.read_text(encoding="utf-8")
        assert "人工复核后的栏目结论" in markdown_text
        assert "Evidence appendix" in markdown_text
        with zipfile.ZipFile(docx_files[0].path) as archive:
            assert "word/document.xml" in archive.namelist()

    # 服务重启：全新的 Store/repository 连接与新 app 实例。
    store.close()
    repository.close()
    store2 = Store(db_path)
    repository2 = ResearchRepository(db_path)
    try:
        app2 = create_app(
            store=store2,
            research_repository=repository2,
            embedder=FakeEmbedder(),
            llm=OfflineLLM(),
        )
        with TestClient(app2) as client:
            assert client.get(f"/api/v1/projects/{project_id}").status_code == 200
            questions = client.get(
                f"/api/v1/projects/{project_id}/questions"
            ).json()
            assert questions["items"]
            detail = client.get(
                f"/api/v1/projects/{project_id}/deep-reads/{artifact_id}"
            ).json()
            assert detail["revision_public"]["created_by"] == "user"
            assert detail["revision_public"]["revision_number"] == 2
            assert detail["freshness"]["stale"] is False
            history = client.get(
                f"/api/v1/projects/{project_id}/deep-reads/{artifact_id}/revisions"
            ).json()
            assert [item["created_by"] for item in history["items"]] == [
                "user",
                "model",
            ]
    finally:
        repository2.close()
        store2.close()


# ---------------------------------------------------------------------------
# 旅程 2：三篇比较与综述
# ---------------------------------------------------------------------------


def _three_paper_project(tmp_path):
    papers = tmp_path / "papers"
    papers.mkdir(exist_ok=True)
    for index in range(1, 4):
        make_pdf(
            papers / f"journey-compare-{index}.pdf",
            [
                "Research question: which citation workflow scales best?",
                "Method: evidence scoped retrieval with bounded generation.",
                "Results: Dataset A shows higher citation accuracy.",
                "Limitations: evaluation covers text only papers.",
            ],
            {"title": f"Journey Compare Paper {index}", "author": "Alice"},
        )
    db_path = tmp_path / "journey2.db"
    store = Store(db_path)
    indexed = index_library(store, papers, FakeEmbedder(), progress=noop_progress)
    assert indexed["added"] == 3

    repository = ResearchRepository(db_path)
    project = repository.create_project("三篇比较旅程")
    repository.create_question(project.id, "哪条引用工作流更好？")
    _, papers_list = store.list_papers(None, 10, 0)
    llm = JourneyLLM()
    source_ids = []
    for paper in papers_list:
        source = repository.ensure_source_for_paper(paper.id)
        repository.add_project_source(project.id, source.id)
        DeepReadArtifactService(repository).generate_and_save(
            project.id, source.id, DeepReadWorkflow(store, FakeEmbedder(), llm)
        )
        source_ids.append(source.id)
    return store, repository, project, source_ids, llm


def test_journey_three_paper_compare_review_style_switch_and_export(tmp_path):
    store, repository, project, source_ids, llm = _three_paper_project(tmp_path)
    try:
        comparison = ComparisonArtifactService(repository).generate_and_save(
            project.id,
            source_ids,
            ComparisonWorkflow(repository),
            title="三篇旅程比较",
        )
        assert comparison.artifact.current_revision_number == 1

        questions = repository.list_questions(project.id)
        outline = ReviewOutlineArtifactService(repository).generate_and_save(
            project.id,
            [questions[0].id],
            source_ids,
            comparison.artifact.id,
            ReviewOutlineWorkflow(repository, llm),
        )
        section = ReviewSectionArtifactService(repository).generate_and_save(
            project.id,
            outline.artifact.id,
            "methods",
            ReviewSectionWorkflow(repository, llm),
        )
        assert section.revision.revision_number == 1

        exports = tmp_path / "journey2-exports"
        service = ArtifactExportService(repository, store)

        gb_files = service.export_current(outline.artifact.id, "markdown", exports)
        gb_snapshot = service.freeze_current(outline.artifact.id)
        assert gb_snapshot.citation_style == "gb-t-7714-2015-numeric"
        gb_markdown = gb_files[0].path.read_text(encoding="utf-8")
        assert "Evidence appendix" in gb_markdown

        updated = repository.update_project(
            project.id,
            expected_version=project.version,
            citation_style="apa-7",
        )
        assert updated.citation_style == "apa-7"
        apa_files = service.export_current(outline.artifact.id, "markdown", exports)
        apa_snapshot = service.freeze_current(outline.artifact.id)
        assert apa_snapshot.citation_style == "apa-7"
        assert apa_files[0].path == gb_files[0].path
        assert apa_files[0].path.read_text(encoding="utf-8") != gb_markdown

        csv_files = service.export_current(comparison.artifact.id, "csv", exports)
        rows = list(
            csv.DictReader(
                io.StringIO(csv_files[0].path.read_text(encoding="utf-8"))
            )
        )
        assert len(rows) == 3 and rows[0]["source_kind"] == "paper"

        json_files = service.export_current(outline.artifact.id, "json", exports)
        envelope = ExportEnvelope.model_validate_json(
            json_files[0].path.read_text(encoding="utf-8")
        )
        assert envelope.revision["id"] == outline.revision.id
        assert len(envelope.sources) == 3

        docx_files = service.export_current(outline.artifact.id, "docx", exports)
        with zipfile.ZipFile(docx_files[0].path) as archive:
            assert "word/document.xml" in archive.namelist()
    finally:
        repository.close()
        store.close()


# ---------------------------------------------------------------------------
# 旅程 3：发现与入库
# ---------------------------------------------------------------------------


class FakeProvider:
    def __init__(self, name, records):
        self.name = name
        self.records = list(records)
        self.calls = []

    def search(self, query, *, limit=10):
        self.calls.append((query, limit))
        return list(self.records)

    def lookup(self, identifier):
        return None


class FixtureFetcher:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def fetch(self, url):
        self.calls.append(url)
        return self.result


WEB_ARTICLE = b"""<!doctype html><html><head>
<title>Journey Web Report</title>
</head><body><article>
<h1>Journey Web Report</h1>
<p>journey web report evidence is available in this public document.</p>
<p>The report keeps a canonical identity for every retrieved source.</p>
</article></body></html>"""


def test_journey_discovery_dedupe_download_webfetch_hybrid_search(tmp_path):
    semantic = NormalizedSource(
        provider="semantic_scholar",
        provider_record_id="Corpus-journey",
        title="Journey Discovery Paper",
        authors=("Alice",),
        year=2025,
        abstract="journey discovery",
        doi="10.1000/journey",
        arxiv_id="2501.00001v2",
        canonical_url="https://semanticscholar.org/paper/journey",
        pdf_url="https://arxiv.org/pdf/2501.00001",
        metadata={"paperId": "Corpus-journey"},
    )
    crossref = NormalizedSource(
        provider="crossref",
        provider_record_id="10.1000/journey",
        title="Journey Discovery Paper",
        authors=("Alice",),
        year=2025,
        doi="https://doi.org/10.1000/JOURNEY",
        canonical_url="https://doi.org/10.1000/journey",
        metadata={"DOI": "10.1000/journey"},
    )

    def downloader(url, target_dir):
        return make_pdf(
            target_dir / "2501.00001.pdf",
            ["journey downloaded paper evidence appears in this paper. " * 20],
            {"title": "Journey Discovery Paper", "author": "Alice"},
        )

    store = Store(tmp_path / "journey3.db")
    repository = ResearchRepository(tmp_path / "journey3.db")
    app = create_app(
        store=store,
        research_repository=repository,
        embedder=FakeEmbedder(),
        llm=OfflineLLM(),
        source_providers=[
            FakeProvider("semantic_scholar", [semantic]),
            FakeProvider("crossref", [crossref]),
        ],
        web_fetcher=FixtureFetcher(
            SafeFetchResult(
                requested_url="https://example.org/journey",
                final_url="https://example.org/journey",
                status_code=200,
                content_type="text/html",
                body=WEB_ARTICLE,
                redirect_chain=(),
                resolved_ips=("93.184.216.34",),
            )
        ),
        snapshot_store=SnapshotStore(tmp_path / "snapshots"),
        pdf_downloader=downloader,
        download_directory=tmp_path / "downloads",
    )
    try:
        with TestClient(app) as client:
            project = client.post(
                "/api/v1/projects", json={"title": "发现入库旅程"}
            ).json()
            discovery = client.post(
                "/api/v1/discover/search",
                json={
                    "query": "journey discovery",
                    "providers": ["semantic_scholar", "crossref"],
                    "limit": 5,
                },
            ).json()
            assert len(discovery["items"]) == 1
            item = discovery["items"][0]
            assert item["duplicate_count"] == 1
            assert set(item["providers"]) == {"crossref", "semantic_scholar"}
            source_id = item["source"]["id"]

            selected = client.post(
                f"/api/v1/projects/{project['id']}/sources/{source_id}"
            )
            assert selected.status_code == 201

            downloaded = client.post(
                f"/api/v1/sources/{source_id}/download",
                json={"project_id": project["id"]},
            )
            assert downloaded.status_code == 200
            assert downloaded.json()["source"]["indexed"] is True

            imported = client.post(
                "/api/v1/sources/web",
                json={"url": "https://example.org/journey", "project_id": project["id"]},
            )
            assert imported.status_code == 201
            assert imported.json()["source"]["indexed"] is True

            members = client.get(
                f"/api/v1/projects/{project['id']}/sources"
            ).json()
            assert members["total"] == 2

        hits = hybrid_search(store, FakeEmbedder(), "journey evidence", top=8)
        kinds = {hit.source_kind for hit in hits}
        assert {"pdf", "web"}.issubset(kinds)
    finally:
        repository.close()
        store.close()


# ---------------------------------------------------------------------------
# 旅程 4：新鲜度与恢复
# ---------------------------------------------------------------------------


def test_journey_staleness_job_restart_and_agent_pending(tmp_path, monkeypatch):
    # -- Part A：重新索引 → stale → 旧版本可读 → 重生成新 revision。
    papers = _index_paper(
        tmp_path,
        "journey-stale.pdf",
        [
            "Stale journey method: evidence scoped retrieval stays bounded.",
            "Stale journey results: Dataset A accuracy improves clearly.",
        ],
        "Journey Stale Paper",
    )
    db_path = tmp_path / "journey4.db"
    store = Store(db_path)
    repository = ResearchRepository(db_path)
    llm = JourneyLLM()
    try:
        index_library(store, papers, FakeEmbedder(), progress=noop_progress)
        _, papers_list = store.list_papers(None, 10, 0)
        paper = papers_list[0]
        project = repository.create_project("新鲜度旅程")
        source = repository.ensure_source_for_paper(paper.id)
        repository.add_project_source(project.id, source.id)
        first = DeepReadArtifactService(repository).generate_and_save(
            project.id, source.id, DeepReadWorkflow(store, FakeEmbedder(), llm)
        )
        assert repository.artifact_freshness(first.artifact.id).stale is False

        make_pdf(
            papers / "journey-stale.pdf",
            ["Rewritten stale journey content changes the source fingerprint."],
            {"title": "Journey Stale Paper", "author": "Alice"},
        )
        reindexed = index_library(
            store, papers, FakeEmbedder(), progress=noop_progress
        )
        assert reindexed["updated"] == 1
        freshness = repository.artifact_freshness(first.artifact.id)
        assert freshness.stale is True

        old_revisions = repository.list_artifact_revisions(first.artifact.id).items
        assert len(old_revisions) == 1
        old_card = DeepReadCard.model_validate(old_revisions[0].content)
        assert old_card is not None

        second = DeepReadArtifactService(repository).generate_and_save(
            project.id, source.id, DeepReadWorkflow(store, FakeEmbedder(), llm)
        )
        assert second.revision.revision_number == 2
        assert repository.artifact_freshness(second.artifact.id).stale is False
    finally:
        repository.close()
        store.close()

    # -- Part B：任务执行中重启 → interrupted → 幂等重排后成功。
    jobs_db = tmp_path / "journey4-jobs.db"
    jobs = JobRepository(jobs_db)
    executed = []

    def handler(context, payload):
        executed.append(payload["n"])
        context.report_progress(1, 1)
        return {"done": payload["n"]}

    try:
        queue = JobQueue(jobs)
        job = queue.enqueue(
            "journey_recover",
            {"n": 1},
            max_attempts=2,
            idempotent=True,
            idempotency_key="journey:recover:1",
            timeout_seconds=30,
        )
        dead = queue.claim("dead-worker", lease_seconds=60)
        assert dead is not None and dead.status == "running"

        report = JobQueue(jobs).recover_startup()
        assert report.interrupted == 1
        assert report.requeued == 1

        pool = WorkerPool(
            JobQueue(jobs),
            {"journey_recover": handler},
            worker_count=1,
            poll_interval=0.01,
        )
        pool.start()
        try:
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                current = jobs.get(job.id)
                if current.status == "succeeded":
                    break
                time.sleep(0.02)
            else:
                raise AssertionError("requeued job did not finish")
        finally:
            pool.stop()
        assert current.status == "succeeded"
        assert executed == [1]
    finally:
        jobs.close()

    # -- Part C：Agent 待确认操作跨重启恢复。
    executed_writes = []

    def write_handler(ctx, text=""):
        executed_writes.append(text)
        return ToolResult.success(message=f"已写入 {len(text)} 字符")

    register_tool(
        ToolSpec(
            name="journey_fake_write",
            description="旅程测试写入工具",
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
            handler=write_handler,
            effects=frozenset({ToolEffect.WRITE_LOCAL}),
            timeout_seconds=5.0,
            idempotent=True,
        )
    )
    try:
        from helpers import StreamingScriptLLM

        agent_db = tmp_path / "journey4-agent.db"
        store1 = Store(agent_db)
        llm1 = StreamingScriptLLM(
            [
                {
                    "content": "需要写入文件。",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "name": "journey_fake_write",
                            "arguments": {"text": "hello"},
                        }
                    ],
                },
                {"content": "写好了。", "tool_calls": [], "deltas": ["写好了"]},
            ]
        )
        app1 = create_app(store=store1, embedder=FakeEmbedder(), llm=llm1)
        with TestClient(app1) as client:
            first_turn = client.post(
                "/api/agent/chat",
                json={"session_id": "journey-agent", "question": "写入资料"},
            )
            assert first_turn.status_code == 200
            assert '"type": "pending"' in first_turn.text
            assert executed_writes == []
        store1.close()

        store2 = Store(agent_db)
        try:
            app2 = create_app(
                store=store2,
                embedder=FakeEmbedder(),
                llm=StreamingScriptLLM(
                    [{"content": "写好了。", "tool_calls": [], "deltas": ["写好了"]}]
                ),
            )
            with TestClient(app2) as client:
                restored = client.get("/api/agent/sessions/journey-agent")
                assert restored.status_code == 200
                assert "journey_fake_write" in restored.text
                confirmed = client.post(
                    "/api/agent/confirm",
                    json={"session_id": "journey-agent", "confirm": True},
                )
                assert confirmed.status_code == 200
                assert '"code": "confirmed"' in confirmed.text
                assert "写好了" in confirmed.text
            assert executed_writes == ["hello"]
            assert store2.list_agent_runs()[0].status == "succeeded"
        finally:
            store2.close()
    finally:
        unregister_tool("journey_fake_write")
