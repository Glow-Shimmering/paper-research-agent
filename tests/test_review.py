import json
import time

import pytest
from fastapi.testclient import TestClient

from helpers import make_paper
from pragent.jobs import JobQueue
from pragent.models import Chunk
from pragent.research import (
    DEEP_READ_FIELD_ORDER,
    ComparisonArtifactService,
    ComparisonWorkflow,
    DeepReadCard,
    ReviewOutline,
    ReviewOutlineArtifactService,
    ReviewOutlinePrerequisiteError,
    ReviewOutlineSchemaError,
    ReviewOutlineWorkflow,
)
from pragent.storage import ArtifactValidationError, JobRepository, ResearchRepository
from pragent.store import Store
from pragent.webapp import create_app


QUOTE = "Exact review evidence sentence."


class ScriptedReviewLLM:
    model = "scripted-review-outline"
    is_configured = True

    def __init__(self, *, invalid_first=False, forged=False):
        self.invalid_first = invalid_first
        self.forged = forged
        self.calls = []
        self._valid = None

    def chat_with_metadata(self, system, user):
        self.calls.append((system, user))
        if "修复下列 JSON" in system:
            content = json.dumps(self._valid, ensure_ascii=False)
        else:
            payload = json.loads(user)
            comparison = payload["comparison"]
            source_ids = comparison["source_ids"]
            refs = []
            for source_id in source_ids:
                cell = next(
                    item
                    for item in comparison["cells"]
                    if item["source_id"] == source_id
                    and item["evidence_refs"]
                )
                refs.append(
                    {
                        "source_id": source_id,
                        **cell["evidence_refs"][0],
                    }
                )
            if self.forged:
                refs[0]["evidence_id"] = refs[1]["evidence_id"]
            claim = {
                "text": "比较两篇论文在方法与结果上的共同点。",
                "source_ids": source_ids,
                "evidence_refs": refs,
                "insufficient_evidence": False,
            }
            self._valid = {
                "title": "证据约束的综述提纲",
                "sections": [
                    {
                        "key": "methods",
                        "title": "方法比较",
                        "objective": "回答方法差异",
                        "source_ids": source_ids,
                        "planned_claims": [claim],
                    },
                    {
                        "key": "results",
                        "title": "结果与局限",
                        "objective": "比较结果边界",
                        "source_ids": source_ids,
                        "planned_claims": [claim],
                    },
                ],
            }
            content = (
                "not-json"
                if self.invalid_first and len(self.calls) == 1
                else json.dumps(self._valid, ensure_ascii=False)
            )
        return {
            "content": content,
            "metadata": {
                "usage": {"total_tokens": 7},
                "finish_reason": "stop",
                "response_id": f"review-{len(self.calls)}",
            },
        }


def _card(evidence_id):
    field = {
        "text": "基于精读卡的综述输入",
        "evidence_refs": [{"evidence_id": evidence_id, "quote": QUOTE}],
        "insufficient_evidence": False,
    }
    return DeepReadCard.model_validate({name: field for name in DEEP_READ_FIELD_ORDER})


def _seed(tmp_path):
    store = Store(tmp_path / "review.db")
    repository = ResearchRepository(store.db_path)
    project = repository.create_project("综述项目")
    question = repository.create_question(project.id, "两类方法如何权衡效果与成本？")
    source_ids = []
    for index in range(2):
        paper_id = store.upsert_paper(
            make_paper(
                f"review-{index}.pdf",
                title=f"Review Paper {index}",
                sha256=f"{index + 11:064x}",
            )
        )
        store.replace_chunks(
            paper_id,
            [
                Chunk(None, paper_id, 0, 1, f"{QUOTE} Source {index}.", None),
                Chunk(
                    None,
                    paper_id,
                    1,
                    2,
                    f"Extra project evidence {index} not used by comparison.",
                    None,
                ),
            ],
        )
        evidence = store.pin_evidence(store.paper_chunks(paper_id)[0].id)
        source = repository.ensure_source_for_paper(paper_id)
        repository.add_project_source(project.id, source.id, position=index)
        deep_read = repository.create_artifact(
            project.id,
            "deep_read",
            source_id=source.id,
            title=f"Review Paper {index} · 精读卡",
            status="generating",
        )
        repository.append_validated_deep_read_revision(
            deep_read.id,
            _card(evidence.id).model_dump(mode="json"),
            expected_artifact_version=deep_read.version,
            expected_source_fingerprint=repository.artifact_freshness(
                deep_read.id
            ).current_fingerprint,
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
    comparison = ComparisonArtifactService(repository).generate_and_save(
        project.id,
        source_ids,
        ComparisonWorkflow(repository),
    )
    return store, repository, project, question, source_ids, comparison


def _revision_refs(outline: ReviewOutline):
    refs = []
    for section in outline.sections:
        for claim_index, claim in enumerate(section.planned_claims):
            field_path = f"$.sections.{section.key}.claims.{claim_index}"
            for ordinal, ref in enumerate(claim.evidence_refs):
                refs.append(
                    (
                        ref.evidence_id,
                        field_path,
                        ordinal,
                        ref.source_id,
                        ref.quote,
                    )
                )
    return refs


def test_review_outline_uses_questions_sources_and_current_comparison(tmp_path):
    store, repository, project, question, source_ids, comparison = _seed(tmp_path)
    workflow = ReviewOutlineWorkflow(repository, ScriptedReviewLLM())

    draft = workflow.generate(
        project.id, [question.id], source_ids, comparison.artifact.id
    )

    assert draft.outline.research_questions[0].question == question.question
    assert draft.outline.source_ids == source_ids
    assert draft.outline.comparison_revision_id == comparison.revision.id
    assert len(draft.outline.sections) == 2
    assert draft.usage["llm_calls"] == 1
    assert draft.model == "scripted-review-outline"
    with pytest.raises(ReviewOutlinePrerequisiteError, match="一致"):
        workflow.generate(
            project.id,
            [question.id],
            list(reversed(source_ids)),
            comparison.artifact.id,
        )
    repository.close()
    store.close()


def test_review_outline_repairs_once_and_rejects_forged_evidence(tmp_path):
    store, repository, project, question, source_ids, comparison = _seed(tmp_path)
    repaired = ReviewOutlineWorkflow(
        repository, ScriptedReviewLLM(invalid_first=True)
    ).generate(project.id, [question.id], source_ids, comparison.artifact.id)
    assert repaired.usage["llm_calls"] == 2
    assert repaired.usage["repair_used"] is True
    assert repaired.usage["total_tokens"] == 14

    with pytest.raises(ReviewOutlineSchemaError, match="越界 evidence|改写 quote"):
        ReviewOutlineWorkflow(
            repository, ScriptedReviewLLM(forged=True)
        ).generate(project.id, [question.id], source_ids, comparison.artifact.id)
    repository.close()
    store.close()


def test_review_outline_artifact_saves_auditable_revision(tmp_path):
    store, repository, project, question, source_ids, comparison = _seed(tmp_path)
    saved = ReviewOutlineArtifactService(repository).generate_and_save(
        project.id,
        [question.id],
        source_ids,
        comparison.artifact.id,
        ReviewOutlineWorkflow(repository, ScriptedReviewLLM()),
    )

    outline = ReviewOutline.model_validate(saved.revision.content)
    assert saved.artifact.artifact_type == "review_outline"
    assert saved.revision.created_by == "model"
    assert saved.revision.model == "scripted-review-outline"
    assert saved.revision.prompt_version == "review-outline-v1"
    assert outline.comparison_revision_id == comparison.revision.id
    assert len(repository.list_artifact_evidence(saved.revision.id)) == 4
    repository.close()
    store.close()


def test_review_outline_atomic_save_rejects_changed_question(tmp_path):
    store, repository, project, question, source_ids, comparison = _seed(tmp_path)
    draft = ReviewOutlineWorkflow(repository, ScriptedReviewLLM()).generate(
        project.id, [question.id], source_ids, comparison.artifact.id
    )
    artifact = repository.create_artifact(
        project.id, "review_outline", status="generating"
    )
    repository.update_question(
        question.id,
        project_id=project.id,
        expected_version=question.version,
        question="已经改变的研究问题",
    )

    with pytest.raises(ArtifactValidationError, match="研究问题已变化"):
        repository.append_validated_review_outline_revision(
            artifact.id,
            draft.outline.model_dump(mode="json"),
            expected_artifact_version=artifact.version,
            expected_project_fingerprint=repository.project_source_fingerprint(
                project.id
            ),
            question_snapshots=[
                (item.id, item.version, item.question)
                for item in draft.outline.research_questions
            ],
            selected_source_ids=source_ids,
            comparison_artifact_id=comparison.artifact.id,
            comparison_revision_id=comparison.revision.id,
            evidence_refs=_revision_refs(draft.outline),
            created_by="model",
            model=draft.model,
            usage=draft.usage,
            finish_reason=draft.finish_reason,
            prompt_version=draft.prompt_version,
            schema_version=draft.schema_version,
        )
    assert repository.get_artifact(artifact.id).current_revision_number == 0
    repository.close()
    store.close()


def test_review_outline_atomic_save_rejects_evidence_outside_comparison(tmp_path):
    store, repository, project, question, source_ids, comparison = _seed(tmp_path)
    draft = ReviewOutlineWorkflow(repository, ScriptedReviewLLM()).generate(
        project.id, [question.id], source_ids, comparison.artifact.id
    )
    source = repository.get_source(source_ids[0])
    extra = store.pin_evidence(store.paper_chunks(source.indexed_paper_id)[1].id)
    content = draft.outline.model_dump(mode="json")
    first_ref = content["sections"][0]["planned_claims"][0]["evidence_refs"][0]
    first_ref["evidence_id"] = extra.id
    first_ref["quote"] = "Extra project evidence 0"
    forged = ReviewOutline.model_validate(content)
    artifact = repository.create_artifact(
        project.id, "review_outline", status="generating"
    )

    with pytest.raises(ArtifactValidationError, match="不在绑定的比较矩阵"):
        repository.append_validated_review_outline_revision(
            artifact.id,
            forged.model_dump(mode="json"),
            expected_artifact_version=artifact.version,
            expected_project_fingerprint=repository.project_source_fingerprint(
                project.id
            ),
            question_snapshots=[
                (item.id, item.version, item.question)
                for item in forged.research_questions
            ],
            selected_source_ids=source_ids,
            comparison_artifact_id=comparison.artifact.id,
            comparison_revision_id=comparison.revision.id,
            evidence_refs=_revision_refs(forged),
            created_by="model",
            model=draft.model,
            usage=draft.usage,
            finish_reason=draft.finish_reason,
            prompt_version=draft.prompt_version,
            schema_version=draft.schema_version,
        )
    assert repository.get_artifact(artifact.id).current_revision_number == 0
    repository.close()
    store.close()


def test_review_outline_runs_through_persistent_worker(tmp_path):
    store, repository, project, question, source_ids, comparison = _seed(tmp_path)
    jobs = JobRepository(store.db_path)
    queued = JobQueue(jobs).enqueue(
        "review_outline",
        {
            "project_id": project.id,
            "question_ids": [question.id],
            "source_ids": source_ids,
            "comparison_artifact_id": comparison.artifact.id,
            "title": "后台综述提纲",
        },
        project_id=project.id,
        idempotent=True,
        idempotency_key=f"review-outline:{project.id}",
        timeout_seconds=30,
    )
    app = create_app(
        store=store,
        research_repository=repository,
        job_repository=jobs,
        llm=ScriptedReviewLLM(),
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
            raise AssertionError("review outline job did not finish")

    assert current.status == "succeeded"
    artifact = repository.get_artifact(current.result["artifact_id"])
    assert artifact is not None and artifact.artifact_type == "review_outline"
    jobs.close()
    repository.close()
    store.close()
