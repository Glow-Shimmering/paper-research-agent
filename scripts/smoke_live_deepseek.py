"""Live DeepSeek smoke：用真实模型跑研究工作流并输出脱敏证据记录。

需要用户显式授权与密钥后手动运行（roadmap 规则 6）：

    export PRA_LLM_API_KEY=...            # 由用户提供；脚本不读取任何密钥文件
    python scripts/smoke_live_deepseek.py --pdf paper1.pdf --pdf paper2.pdf --pdf paper3.pdf

行为：
1. 在临时数据目录索引给定 PDF（真实 Embedder；首次运行会联网下载嵌入模型）；
2. 对每篇 PDF 生成九栏精读卡（真实 DeepSeek，走 DeepReadWorkflow 的
   字段检索 → map/reduce → 校验 → 原子保存完整合同）；
3. 以全部来源生成比较矩阵（默认九维复用精读卡，不额外调用模型）；
4. 生成综述提纲与第一节草稿（结构化 citation tokens + scope/quote 校验）；
5. 输出脱敏 JSON 证据：模型名、usage、finish_reason、耗时、revision 数、
   schema/prompt version、失败/错误码；不输出论文正文、quote 原文或密钥。

不带密钥时 fail closed，不访问网络。退出码：0 = 全部步骤成功。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

from pragent.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
from pragent.embeddings import Embedder
from pragent.indexer import index_library
from pragent.llm import LLMClient
from pragent.research import (
    ComparisonArtifactService,
    ComparisonWorkflow,
    DeepReadArtifactService,
    DeepReadWorkflow,
    ReviewOutlineArtifactService,
    ReviewOutlineWorkflow,
    ReviewSectionArtifactService,
    ReviewSectionWorkflow,
)
from pragent.storage import ResearchRepository
from pragent.store import Store


def _fail(message: str) -> int:
    print(message, file=sys.stderr)
    return 1


def _step(name: str, run) -> dict:
    started = time.monotonic()
    try:
        result = run()
    except Exception as exc:
        return {
            "name": name,
            "ok": False,
            "error_code": str(getattr(exc, "code", exc.__class__.__name__))[:80],
            "error_message": str(exc)[:300],
            "elapsed_seconds": round(time.monotonic() - started, 2),
        }
    return {
        "name": name,
        "ok": True,
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "result": _redacted(result),
    }


def _redacted(result) -> dict:
    """只保留 id/revision/usage/finish_reason 等元数据；不带正文、quote 或密钥。"""
    revision = result.revision
    record = {
        "artifact_id": result.artifact.id,
        "artifact_title": result.artifact.title,
        "revision_number": revision.revision_number,
        "model": revision.model,
        "usage": revision.usage,
        "finish_reason": revision.finish_reason,
        "prompt_version": revision.prompt_version,
        "schema_version": revision.schema_version,
        "created_by": revision.created_by,
    }
    content = revision.content if isinstance(revision.content, dict) else {}
    if isinstance(content.get("sections"), list):
        record["sections"] = [
            {"key": section.get("key"), "title": section.get("title")}
            for section in content["sections"]
            if isinstance(section, dict)
        ]
    return record


def _finish(record: dict, output: Path | None) -> int:
    record["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    record["all_ok"] = all(step.get("ok") for step in record["steps"])
    payload = json.dumps(record, ensure_ascii=False, indent=2)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")
        print(f"证据记录已写入：{output}")
    print(payload)
    print("提示：该记录为脱敏元数据；模型质量的人工检查结论需另行记录。")
    return 0 if record["all_ok"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pdf",
        action="append",
        type=Path,
        required=True,
        help="参与演练的本地 PDF（可重复；比较至少需要 2 篇）",
    )
    parser.add_argument("--json", type=Path, help="把脱敏证据记录写入该 JSON 文件")
    parser.add_argument(
        "--skip-review", action="store_true", help="只跑精读与比较，跳过提纲/章节"
    )
    args = parser.parse_args()

    if not LLM_API_KEY:
        return _fail(
            "未配置 PRA_LLM_API_KEY：live smoke 需要用户显式授权并提供密钥；"
            "脚本不读取密钥文件，也不会在无密钥时访问网络。"
        )

    pdfs = [path.expanduser().resolve() for path in args.pdf]
    for path in pdfs:
        if not path.is_file():
            return _fail(f"PDF 不存在：{path}")
    if len(pdfs) < 2:
        return _fail("live smoke 至少需要 2 篇 PDF（比较矩阵要求 2–20 个来源）")

    record: dict = {
        "kind": "pragent-live-deepseek-smoke",
        "model": LLM_MODEL,
        "base_url_host": LLM_BASE_URL,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "steps": [],
    }
    llm = LLMClient(LLM_BASE_URL, LLM_API_KEY, LLM_MODEL)
    embedder = Embedder()

    with tempfile.TemporaryDirectory(prefix="pra-live-smoke-") as raw:
        root = Path(raw)
        papers_dir = root / "papers"
        papers_dir.mkdir()
        for path in pdfs:
            shutil.copy2(path, papers_dir / path.name)

        store = Store(root / "library.db")
        repository = ResearchRepository(root / "library.db")
        try:
            step = _step(
                "index",
                lambda: index_library(
                    store, papers_dir, embedder, progress=lambda _m: None
                ),
            )
            record["steps"].append(step)
            if not step["ok"]:
                return _finish(record, args.json)

            _, papers = store.list_papers(None, 100, 0)
            if len(papers) < len(pdfs):
                step["ok"] = False
                step["error_code"] = "index_incomplete"
                step["error_message"] = (
                    f"仅索引到 {len(papers)} 篇（预期 {len(pdfs)}）；"
                    "请确认 PDF 含可抽取文本"
                )
                return _finish(record, args.json)

            project = repository.create_project("Live DeepSeek smoke")
            source_ids = []
            for paper in papers:
                source = repository.ensure_source_for_paper(paper.id)
                repository.add_project_source(project.id, source.id)
                source_ids.append(source.id)
                record["steps"].append(
                    _step(
                        f"deep_read:{(paper.title or paper.id)[:40]}",
                        lambda source_id=source.id: DeepReadArtifactService(
                            repository
                        ).generate_and_save(
                            project.id,
                            source_id,
                            DeepReadWorkflow(store, embedder, llm),
                        ),
                    )
                )
                if not record["steps"][-1]["ok"]:
                    return _finish(record, args.json)

            record["steps"].append(
                _step(
                    "comparison",
                    lambda: ComparisonArtifactService(repository).generate_and_save(
                        project.id,
                        source_ids,
                        ComparisonWorkflow(repository),
                    ),
                )
            )
            if not record["steps"][-1]["ok"]:
                return _finish(record, args.json)

            if not args.skip_review:
                questions = repository.list_questions(project.id)
                if not questions.total:
                    repository.create_question(
                        project.id, "这些论文的共同点是什么？"
                    )
                    questions = repository.list_questions(project.id)
                comparison_artifact_id = record["steps"][-1]["result"]["artifact_id"]

                record["steps"].append(
                    _step(
                        "review_outline",
                        lambda: ReviewOutlineArtifactService(
                            repository
                        ).generate_and_save(
                            project.id,
                            [questions.items[0].id],
                            source_ids,
                            comparison_artifact_id,
                            ReviewOutlineWorkflow(repository, llm),
                        ),
                    )
                )
                if not record["steps"][-1]["ok"]:
                    return _finish(record, args.json)

                outline_result = record["steps"][-1]["result"]
                section_key = (outline_result.get("sections") or [{}])[0].get("key")
                if not section_key:
                    record["steps"][-1]["ok"] = False
                    record["steps"][-1]["error_code"] = "outline_without_sections"
                    return _finish(record, args.json)

                record["steps"].append(
                    _step(
                        f"review_section:{section_key}",
                        lambda: ReviewSectionArtifactService(
                            repository
                        ).generate_and_save(
                            project.id,
                            outline_result["artifact_id"],
                            section_key,
                            ReviewSectionWorkflow(repository, llm),
                        ),
                    )
                )
        finally:
            repository.close()
            store.close()

    return _finish(record, args.json)


if __name__ == "__main__":
    sys.exit(main())
