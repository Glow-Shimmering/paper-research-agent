"""Step 28 数据备份恢复演练：备份 PRAgent 数据目录后在空目录恢复并验证。

离线、确定性、无真实 key。模拟完整生命周期：

1. 在临时「生产」数据目录按真实布局构建数据（``library.db`` + ``snapshots/``）：
   索引一篇本地 PDF、创建项目/研究问题/来源、生成九栏精读卡（脚本化 LLM）、
   保存笔记、抓取并索引一个网页快照、运行一个持久任务到 succeeded；
2. 以文件复制方式备份整个数据目录，并校验前后 SHA-256 清单一致；
3. 把备份恢复到另一个全新空目录（模拟数据盘丢失后在空目录还原）；
4. 在恢复目录上重新打开 Store/ResearchRepository，逐项验证：
   项目/问题/来源 membership、精读卡当前 revision 与 evidence links、笔记、
   succeeded 任务、snapshot 文件与 SHA-256、以及 hybrid search 可用；
5. 全程不读写真实的 ``~/.pragent`` 与 ``~/.pagent``。

用法：
    python scripts/backup_restore_drill.py            # 使用临时目录
    python scripts/backup_restore_drill.py --keep     # 保留演练目录供人工检查
退出码：0 = 演练通过；1 = 存在验证失败。
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import fitz
import numpy as np

from pragent.ingestion.indexing import index_web_source
from pragent.ingestion.safe_fetch import SafeFetchResult
from pragent.ingestion.snapshots import SnapshotStore
from pragent.ingestion.web import WebIngestService
from pragent.indexer import index_library
from pragent.jobs import JobQueue, WorkerPool
from pragent.research import DEEP_READ_FIELD_ORDER, DeepReadArtifactService, DeepReadWorkflow
from pragent.search import hybrid_search
from pragent.storage import JobRepository, ResearchRepository
from pragent.store import Store

DB_NAME = "library.db"
SNAP_DIR_NAME = "snapshots"


class FixtureEmbedder:
    model_name = "backup-drill-fixture"

    def embed(self, texts, batch_size=32):
        rows = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            rows.append(np.frombuffer(digest[:32], dtype=np.uint8).astype(np.float32))
        return np.stack(rows)


class ScriptedLLM:
    model = "backup-drill-scripted-llm"
    is_configured = True

    def chat_with_metadata(self, system, user):
        if "精读助手" in system:
            payload = json.loads(user)
            evidence = payload["evidence"]
            if evidence:
                content = {
                    "text": f"{payload['label']}：演练总结。",
                    "evidence_refs": [
                        {
                            "evidence_id": evidence[0]["evidence_id"],
                            "quote": evidence[0]["text"][:80],
                        }
                    ],
                    "insufficient_evidence": False,
                }
            else:
                content = {
                    "text": "证据不足",
                    "evidence_refs": [],
                    "insufficient_evidence": True,
                }
        else:
            content = json.loads(user)
        return {
            "content": json.dumps(content, ensure_ascii=False),
            "metadata": {
                "usage": {"total_tokens": 10},
                "finish_reason": "stop",
                "response_id": f"drill-{time.monotonic_ns()}",
            },
        }


WEB_BODY = b"""<!doctype html><html><head><title>Drill Report</title></head>
<body><article><h1>Drill Report</h1>
<p>backup drill web evidence is stored as a content addressed gzip snapshot.</p>
<p>The restored snapshot must keep the same sha256 and extracted text.</p>
</article></body></html>"""


def _write_pdf(path: Path) -> None:
    document = fitz.open()
    page = document.new_page()
    text = (
        "Backup drill paper describes evidence grounded retrieval. "
        "The method keeps citations bounded and verifiable. "
    ) * 12
    page.insert_textbox(fitz.Rect(50, 50, 545, 790), text, fontsize=9)
    document.save(path)
    document.close()


def _hash_manifest(root: Path) -> dict[str, str]:
    manifest: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            manifest[path.relative_to(root).as_posix()] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    return manifest


class FixtureFetcher:
    def fetch(self, url: str) -> SafeFetchResult:
        return SafeFetchResult(
            requested_url=url,
            final_url=url,
            status_code=200,
            content_type="text/html",
            body=WEB_BODY,
            redirect_chain=(),
            resolved_ips=("93.184.216.34",),
        )


def build_data_dir(data_dir: Path) -> dict:
    """按真实布局构建一个最小但完整的数据目录，返回恢复验证所需的标识。"""
    (data_dir / "papers").mkdir(parents=True)
    _write_pdf(data_dir / "papers" / "drill-paper.pdf")

    db_path = data_dir / DB_NAME
    store = Store(db_path)
    repository = ResearchRepository(db_path)
    jobs = JobRepository(db_path)
    try:
        indexed = index_library(
            store, data_dir / "papers", FixtureEmbedder(), progress=lambda _m: None
        )
        assert indexed["added"] == 1, indexed
        _, papers = store.list_papers(None, 10, 0)

        project = repository.create_project("备份恢复演练")
        repository.create_question(project.id, "演练问题：证据是否可恢复？")
        source = repository.ensure_source_for_paper(papers[0].id)
        repository.add_project_source(project.id, source.id)
        repository.create_note(
            project.id, scope_kind="project", content_markdown="演练项目级笔记。"
        )

        saved = DeepReadArtifactService(repository).generate_and_save(
            project.id,
            source.id,
            DeepReadWorkflow(store, FixtureEmbedder(), ScriptedLLM()),
        )

        web = WebIngestService(
            repository,
            fetcher=FixtureFetcher(),
            snapshots=SnapshotStore(data_dir / SNAP_DIR_NAME),
        )
        result = web.ingest("https://example.org/drill")
        index_web_source(store, repository, result.source.id, FixtureEmbedder())

        def handler(_context, payload):
            return {"echo": payload["value"]}

        queue = JobQueue(jobs)
        job = queue.enqueue("drill_echo", {"value": 7}, timeout_seconds=30)
        pool = WorkerPool(
            queue, {"drill_echo": handler}, worker_count=1, poll_interval=0.01
        )
        pool.start()
        try:
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                if jobs.get(job.id).status == "succeeded":
                    break
                time.sleep(0.02)
            else:
                raise AssertionError("drill job did not finish")
        finally:
            pool.stop()
        return {
            "project_id": project.id,
            "paper_source_id": source.id,
            "web_source_id": result.source.id,
            "artifact_id": saved.artifact.id,
            "revision_id": saved.revision.id,
            "job_id": job.id,
        }
    finally:
        jobs.close()
        repository.close()
        store.close()


def verify_restored(restore_dir: Path, expected: dict, failures: list[str]) -> None:
    def check(name: str, condition: bool) -> None:
        if not condition:
            failures.append(f"{name}：恢复后缺失或不一致")

    db_path = restore_dir / DB_NAME
    store = Store(db_path)
    repository = ResearchRepository(db_path)
    jobs = JobRepository(db_path)
    try:
        project = repository.get_project(expected["project_id"])
        check("项目", project is not None)
        questions = repository.list_questions(expected["project_id"])
        check("研究问题", len(questions) == 1)
        members = repository.list_project_sources(
            expected["project_id"], limit=50, offset=0
        )
        check("项目来源 membership", members.total == 1)
        notes = repository.list_notes(expected["project_id"])
        check("笔记", notes.total == 1)

        artifact = repository.get_artifact(expected["artifact_id"])
        check("artifact", artifact is not None)
        revision = repository.get_current_artifact_revision(expected["artifact_id"])
        check(
            "当前 revision",
            revision is not None and revision.id == expected["revision_id"],
        )
        links = repository.list_artifact_evidence(expected["revision_id"])
        check(
            "evidence links 覆盖九栏",
            {link.field_path for link in links}
            == {f"$.{name}" for name in DEEP_READ_FIELD_ORDER},
        )
        freshness = repository.artifact_freshness(expected["artifact_id"])
        check("freshness 计算", freshness.stale is False)

        job = jobs.get(expected["job_id"])
        check("succeeded 任务", job is not None and job.status == "succeeded")

        web_source = repository.get_source(expected["web_source_id"])
        paper_source = repository.get_source(expected["paper_source_id"])
        check("web source ready", web_source is not None and web_source.status == "ready")
        check(
            "paper source ready",
            paper_source is not None and paper_source.status == "ready",
        )

        snapshot_file = restore_dir / SNAP_DIR_NAME / (web_source.snapshot_path or "")
        check("snapshot 文件存在", snapshot_file.is_file())
        if snapshot_file.is_file():
            raw = gzip.decompress(snapshot_file.read_bytes())
            check(
                "snapshot sha256",
                hashlib.sha256(raw).hexdigest() == web_source.snapshot_sha256,
            )

        hits = hybrid_search(
            store, FixtureEmbedder(), "evidence grounded retrieval", top=5
        )
        check("恢复后 hybrid search", len(hits) > 0)
    finally:
        jobs.close()
        repository.close()
        store.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep", action="store_true", help="保留演练目录供人工检查")
    args = parser.parse_args()

    started = time.monotonic()
    print("== 数据备份恢复演练（离线确定性）==")
    if args.keep:
        root = Path(tempfile.gettempdir()) / f"pra-backup-drill-{time.monotonic_ns()}"
        root.mkdir(parents=True)
        print(f"演练根目录（保留）：{root}")
    else:
        holder = tempfile.TemporaryDirectory(prefix="pra-backup-drill-")
        root = Path(holder.name)

    try:
        data_dir = root / "pragent"
        backup_dir = root / "pragent-backup"
        restore_dir = root / "restored-pragent"

        print("1) 构建生产数据目录 …")
        expected = build_data_dir(data_dir)

        print("2) 备份并校验 SHA-256 清单 …")
        shutil.copytree(data_dir, backup_dir)
        if _hash_manifest(data_dir) != _hash_manifest(backup_dir):
            print("   [FAIL] 备份清单与源数据不一致")
            return 1
        print(f"   备份文件数：{len(_hash_manifest(backup_dir))}")

        print("3) 恢复到全新空目录 …")
        shutil.copytree(backup_dir, restore_dir)

        print("4) 在恢复目录上验证项目/来源/artifact/evidence/job/snapshot/索引 …")
        failures: list[str] = []
        verify_restored(restore_dir, expected, failures)
        if failures:
            for failure in failures:
                print(f"   [FAIL] {failure}")
            print("演练未通过")
            return 1

        print(
            "演练通过：备份后可在空目录恢复项目、来源、精读卡与 evidence、"
            "笔记、任务记录、snapshot 与全文索引；"
            "全程未读写真实的 ~/.pragent 与 ~/.pagent。"
        )
        print(f"耗时 {time.monotonic() - started:.1f}s")
        return 0
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
