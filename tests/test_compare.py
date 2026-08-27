import json
import time

import pytest
from fastapi.testclient import TestClient

from helpers import make_paper
from pragent.models import Chunk
from pragent.jobs import JobQueue
from pragent.research import (
    DEEP_READ_FIELD_ORDER,
    ComparisonArtifactService,
    ComparisonDimension,
    ComparisonMatrix,
    ComparisonPrerequisiteError,
    ComparisonSchemaError,
    ComparisonWorkflow,
    DeepReadCard,
)
from pragent.storage import ArtifactValidationError, JobRepository, ResearchRepository
from pragent.store import Store
from pragent.webapp import create_app


QUOTE = "Exact evidence sentence."


class ScriptedComparisonLLM:
    model = "scripted-comparison"

    def __init__(self, *, invalid_first=False, forged=False, altered_quote=False):
        self.invalid_first = invalid_first
        self.forged = forged
        self.altered_quote = altered_quote
        self.calls = []
        self._valid = None

    def chat_with_metadata(self, system, user):
        self.calls.append((system, user))
        if "修复下列 JSON" in system:
            content = json.dumps(self._valid, ensure_ascii=False)
        else:
            payload = json.loads(user)
            dimension = payload["dimension"]
            sources = payload["sources"]
            cells = []
            for source in sources:
                first_field = next(iter(source["deep_read"].values()))
                ref = dict(first_field["evidence_refs"][0])
                cells.append(
                    {
                        "source_id": source["source_id"],
                        "dimension_key": dimension["key"],
                        "summary": f"{source['title']} 的自定义比较结论",
                        "evidence_refs": [ref],
                        "insufficient_evidence": False,
                    }
                )
            if self.forged:
                cells[0]["evidence_refs"] = cells[1]["evidence_refs"]
            if self.altered_quote:
                cells[0]["evidence_refs"][0]["quote"] = "rewritten quote"
            self._valid = {"dimension_key": dimension["key"], "cells": cells}
            content = "not-json" if self.invalid_first and len(self.calls) == 1 else json.dumps(
                self._valid, ensure_ascii=False
            )
        return {
            "content": content,
            "metadata": {
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
                "finish_reason": "stop",
                "response_id": f"cmp-{len(self.calls)}",
            },
        }


def _card(evidence_id):
    field = {
        "text": "基于精读卡的结论",
        "evidence_refs": [{"evidence_id": evidence_id, "quote": QUOTE}],
        "insufficient_evidence": False,
    }
    return DeepReadCard.model_validate({name: field for name in DEEP_READ_FIELD_ORDER})


def _seed_project(tmp_path, count=2):
    store = Store(tmp_path / "compare.db")
    repository = ResearchRepository(store.db_path)
    project = repository.create_project("比较项目")
    source_ids = []
    evidence_ids = []
    paper_ids = []
    for index in range(count):
        path = f"paper-{index}.pdf"
        paper_id = store.upsert_paper(
            make_paper(
                path,
                title=f"Paper {index}",
                sha256=f"{index + 1:064x}",
            )
        )
        store.replace_chunks(
            paper_id,
            [Chunk(None, paper_id, 0, 1, f"{QUOTE} Source {index}.", None)],
        )
        evidence = store.pin_evidence(store.paper_chunks(paper_id)[0].id)
        source = repository.ensure_source_for_paper(paper_id)
        repository.add_project_source(project.id, source.id, position=index)
        artifact = repository.create_artifact(
            project.id,
            "deep_read",
            source_id=source.id,
            title=f"Paper {index} · 精读卡",
            status="generating",
        )
        fingerprint = repository.artifact_freshness(artifact.id).current_fingerprint
        repository.append_validated_deep_read_revision(
            artifact.id,
            _card(evidence.id).model_dump(mode="json"),
            expected_artifact_version=artifact.version,
            expected_source_fingerprint=fingerprint,
            created_by="model",
            evidence_refs=[
                (evidence.id, f"$.{name}", 0, QUOTE)
                for name in DEEP_READ_FIELD_ORDER
            ],
            model="scripted-deep-read",
            usage={"total_tokens": 1},
            finish_reason="stop",
            prompt_version="deep-read-v1",
            schema_version=1,
        )
        source_ids.append(source.id)
        evidence_ids.append(evidence.id)
        paper_ids.append(paper_id)
    return store, repository, project, source_ids, evidence_ids, paper_ids


def _custom_dimension():
    return ComparisonDimension(
        key="deployment_cost",
        label="部署成本",
        description="比较论文明确报告的部署资源与成本。",
    )


def test_comparison_matrix_requires_complete_cartesian_shape():
    dimensions = [
        ComparisonDimension(key="method", label="方法"),
        ComparisonDimension(key="results", label="结果"),
    ]
    cells = [
        {
            "source_id": source_id,
            "dimension_key": "method",
            "summary": "证据不足",
            "evidence_refs": [],
            "insufficient_evidence": True,
        }
        for source_id in ("source-a", "source-b")
    ]
    with pytest.raises(ValueError, match="完整覆盖"):
        ComparisonMatrix(
            title="矩阵",
            source_ids=["source-a", "source-b"],
            dimensions=dimensions,
            cells=cells,
        )


def test_comparison_is_project_scoped_and_requires_two_current_deep_reads(tmp_path):
    store, repository, project, source_ids, _, _ = _seed_project(tmp_path, count=2)
    workflow = ComparisonWorkflow(repository)

    with pytest.raises(ValueError, match="2–20"):
        workflow.generate(project.id, source_ids[:1])
    with pytest.raises(ValueError, match="属于当前项目"):
        workflow.generate(project.id, [source_ids[0], "source-outside"])

    third_paper = store.upsert_paper(
        make_paper("third.pdf", title="Third", sha256="f" * 64)
    )
    third_source = repository.ensure_source_for_paper(third_paper)
    repository.add_project_source(project.id, third_source.id)
    with pytest.raises(ComparisonPrerequisiteError) as exc_info:
        workflow.generate(project.id, [source_ids[0], third_source.id])
    assert exc_info.value.missing_source_ids == (third_source.id,)
    repository.close()
    store.close()


def test_default_comparison_reuses_deep_read_without_llm(tmp_path):
    store, repository, project, source_ids, evidence_ids, _ = _seed_project(tmp_path)

    draft = ComparisonWorkflow(repository).generate(project.id, source_ids)

    assert len(draft.matrix.dimensions) == 9
    assert len(draft.matrix.cells) == 18
    assert draft.usage["llm_calls"] == 0
    assert draft.model is None
    assert {
        cell.evidence_refs[0].evidence_id for cell in draft.matrix.cells
    } == set(evidence_ids)
    repository.close()
    store.close()


def test_custom_dimension_is_one_bounded_call_and_repairs_once(tmp_path):
    store, repository, project, source_ids, _, _ = _seed_project(tmp_path)
    llm = ScriptedComparisonLLM(invalid_first=True)

    draft = ComparisonWorkflow(repository, llm).generate(
        project.id,
        source_ids,
        custom_dimensions=[_custom_dimension()],
    )

    assert len(draft.matrix.dimensions) == 10
    assert len(draft.matrix.cells) == 20
    assert draft.usage["llm_calls"] == 2
    assert draft.usage["repair_used"] is True
    assert draft.usage["total_tokens"] == 10
    assert draft.model == "scripted-comparison"
    repository.close()
    store.close()


@pytest.mark.parametrize("option", ["forged", "altered_quote"])
def test_custom_dimension_rejects_cross_source_or_rewritten_evidence(tmp_path, option):
    store, repository, project, source_ids, _, _ = _seed_project(tmp_path)
    llm = ScriptedComparisonLLM(**{option: True})

    with pytest.raises(ComparisonSchemaError, match="越界 evidence|改写 quote"):
        ComparisonWorkflow(repository, llm).generate(
            project.id,
            source_ids,
            custom_dimensions=[_custom_dimension()],
        )
    repository.close()
    store.close()


def test_comparison_artifact_atomically_saves_project_evidence_and_metadata(tmp_path):
    store, repository, project, source_ids, _, _ = _seed_project(tmp_path)
    llm = ScriptedComparisonLLM()
    workflow = ComparisonWorkflow(repository, llm)

    saved = ComparisonArtifactService(repository).generate_and_save(
        project.id,
        source_ids,
        workflow,
        custom_dimensions=[_custom_dimension()],
    )

    assert saved.artifact.artifact_type == "comparison"
    assert saved.artifact.source_id is None
    assert saved.artifact.status == "ready"
    assert saved.revision.created_by == "model"
    assert saved.revision.model == "scripted-comparison"
    assert saved.revision.schema_version == 1
    assert repository.artifact_freshness(saved.artifact.id).stale is False
    links = repository.list_artifact_evidence(saved.revision.id)
    assert len(links) == 20
    assert all(link.field_path.startswith("$.cells.source_") for link in links)
    repository.close()
    store.close()


def test_comparison_revision_rejects_cross_source_evidence_atomically(tmp_path):
    store, repository, project, source_ids, evidence_ids, _ = _seed_project(tmp_path)
    draft = ComparisonWorkflow(repository).generate(project.id, source_ids)
    artifact = repository.create_artifact(
        project.id, "comparison", title="越界比较", status="generating"
    )
    fingerprint = repository.artifact_freshness(artifact.id).current_fingerprint
    content = draft.matrix.model_dump(mode="json")
    first = next(cell for cell in content["cells"] if cell["source_id"] == source_ids[0])
    first["evidence_refs"][0]["evidence_id"] = evidence_ids[1]
    refs = []
    for cell in content["cells"]:
        for ordinal, ref in enumerate(cell["evidence_refs"]):
            refs.append(
                (
                    ref["evidence_id"],
                    f"$.cells.{cell['source_id']}.{cell['dimension_key']}",
                    ordinal,
                    cell["source_id"],
                    ref["quote"],
                )
            )

    with pytest.raises(ArtifactValidationError, match="不属于声明来源"):
        repository.append_validated_comparison_revision(
            artifact.id,
            content,
            expected_artifact_version=artifact.version,
            expected_project_fingerprint=fingerprint,
            selected_source_ids=source_ids,
            evidence_refs=refs,
            created_by="system",
            model=None,
            usage={"llm_calls": 0},
            finish_reason=None,
            prompt_version="comparison-v1",
            schema_version=1,
        )
    assert repository.get_artifact(artifact.id).current_revision_number == 0
    repository.close()
    store.close()


def test_comparison_runs_through_persistent_worker_handler(tmp_path):
    store, repository, project, source_ids, _, _ = _seed_project(tmp_path)
    jobs = JobRepository(store.db_path)
    queued = JobQueue(jobs).enqueue(
        "comparison",
        {
            "project_id": project.id,
            "source_ids": source_ids,
            "title": "后台比较",
            "custom_dimensions": [],
        },
        project_id=project.id,
        idempotent=True,
        idempotency_key=f"comparison:{project.id}",
        timeout_seconds=30,
    )
    app = create_app(
        store=store,
        research_repository=repository,
        job_repository=jobs,
        llm=ScriptedComparisonLLM(),
        job_worker_count=1,
    )

    with TestClient(app, base_url="http://127.0.0.1"):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            current = jobs.get(queued.id)
            if current is not None and current.status in {
                "succeeded",
                "failed",
                "cancelled",
            }:
                break
            time.sleep(0.02)
        else:
            raise AssertionError("comparison job did not finish")

    assert current.status == "succeeded"
    artifact = repository.get_artifact(current.result["artifact_id"])
    assert artifact is not None and artifact.status == "ready"
    jobs.close()
    repository.close()
    store.close()
