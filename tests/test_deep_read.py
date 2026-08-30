import json

import pytest

from helpers import FakeEmbedder, make_paper
from pragent.models import Chunk
from pragent.research import (
    DEEP_READ_FIELD_ORDER,
    DeepReadArtifactService,
    DeepReadBudget,
    DeepReadBudgetExceeded,
    DeepReadCard,
    DeepReadField,
    DeepReadSchemaError,
    DeepReadWorkflow,
)
from pragent.storage import ResearchRepository
from pragent.store import Store


class ScriptedDeepReadLLM:
    model = "scripted-deep-read"

    def __init__(self, *, invalid_calls=0, invalid_repair=False, forged_reduce=False):
        self.invalid_calls = invalid_calls
        self.invalid_repair = invalid_repair
        self.forged_reduce = forged_reduce
        self.calls = []
        self.last_evidence_id = None

    def chat_with_metadata(self, system, user):
        self.calls.append((system, user))
        if "修复下列 JSON" in system:
            content = "still invalid" if self.invalid_repair else self._field_json()
        elif "精读助手" in system:
            payload = json.loads(user)
            evidence = payload["evidence"]
            self.last_evidence_id = evidence[0]["evidence_id"] if evidence else None
            if self.invalid_calls:
                self.invalid_calls -= 1
                content = "not-json"
            else:
                content = self._field_json()
        else:
            payload = json.loads(user)
            if self.forged_reduce:
                payload["core_method"]["evidence_refs"][0]["evidence_id"] = "ev_" + "f" * 64
            content = json.dumps(payload, ensure_ascii=False)
        return {
            "content": content,
            "metadata": {
                "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
                "finish_reason": "stop",
                "response_id": f"resp-{len(self.calls)}",
            },
        }

    def _field_json(self):
        if self.last_evidence_id is None:
            value = {
                "text": "证据不足",
                "evidence_refs": [],
                "insufficient_evidence": True,
            }
        else:
            value = {
                "text": "基于原文的中文总结",
                "evidence_refs": [
                    {"evidence_id": self.last_evidence_id, "quote": "Evidence sentence."}
                ],
                "insufficient_evidence": False,
            }
        return json.dumps(value, ensure_ascii=False)


def _store(tmp_path):
    store = Store(tmp_path / "deep.db")
    paper_id = store.upsert_paper(make_paper("deep.pdf", title="Deep Paper"))
    texts = [
        "Evidence sentence. The paper proposes a retrieval method.",
        "Evidence sentence. Experiments use Dataset A and report gains.",
        "Evidence sentence. The authors discuss limitations and future work.",
    ]
    store.replace_chunks(
        paper_id,
        [
            Chunk(None, paper_id, index, index + 1, text, FakeEmbedder.vecs_for(text))
            for index, text in enumerate(texts)
        ],
    )
    return store, paper_id


def test_deep_read_schema_has_exact_nine_ordered_fields():
    insufficient = {
        "text": "证据不足",
        "evidence_refs": [],
        "insufficient_evidence": True,
    }
    card = DeepReadCard.model_validate(
        {name: insufficient for name in DEEP_READ_FIELD_ORDER}
    )
    assert tuple(name for name, _ in card.ordered_fields()) == DEEP_READ_FIELD_ORDER
    with pytest.raises(ValueError, match="evidence_refs"):
        DeepReadField(text="无证据事实", evidence_refs=[])


def test_deep_read_field_retrieval_map_reduce_and_metadata_are_bounded(tmp_path):
    store, paper_id = _store(tmp_path)
    llm = ScriptedDeepReadLLM()
    workflow = DeepReadWorkflow(store, FakeEmbedder(), llm)

    draft = workflow.generate(paper_id)

    assert isinstance(draft.card, DeepReadCard)
    assert len(llm.calls) == 10
    assert draft.usage["llm_calls"] == 10
    assert draft.usage["retrieval_calls"] == 9
    assert draft.usage["total_tokens"] == 50
    assert draft.usage["repair_used"] is False
    assert draft.model == "scripted-deep-read"
    assert draft.finish_reason == "stop"
    assert draft.evidence
    assert all(
        ref.evidence_id in draft.evidence
        for _, field in draft.card.ordered_fields()
        for ref in field.evidence_refs
    )
    store.close()


def test_deep_read_validates_and_atomically_saves_revision_metadata(tmp_path):
    store, paper_id = _store(tmp_path)
    repository = ResearchRepository(store.db_path)
    project = repository.create_project("精读保存")
    source = repository.ensure_source_for_paper(paper_id)
    repository.add_project_source(project.id, source.id)
    workflow = DeepReadWorkflow(store, FakeEmbedder(), ScriptedDeepReadLLM())

    saved = DeepReadArtifactService(repository).generate_and_save(
        project.id, source.id, workflow
    )

    assert saved.artifact.current_revision_number == 1
    assert saved.revision.created_by == "model"
    assert saved.revision.model == "scripted-deep-read"
    assert saved.revision.usage["llm_calls"] == 10
    assert repository.artifact_freshness(saved.artifact.id).stale is False
    links = repository.list_artifact_evidence(saved.revision.id)
    assert {link.field_path for link in links} == {
        f"$.{name}" for name in DEEP_READ_FIELD_ORDER
    }
    repository.close()
    store.close()


def test_deep_read_uses_exactly_one_json_repair(tmp_path):
    store, paper_id = _store(tmp_path)
    llm = ScriptedDeepReadLLM(invalid_calls=1)
    draft = DeepReadWorkflow(store, FakeEmbedder(), llm).generate(paper_id)
    assert draft.usage["repair_used"] is True
    assert draft.usage["llm_calls"] == 11
    assert "修复下列 JSON" in llm.calls[1][0]
    store.close()


def test_deep_read_rejects_failed_repair_and_forged_reduce_evidence(tmp_path):
    store, paper_id = _store(tmp_path)
    broken = ScriptedDeepReadLLM(invalid_calls=1, invalid_repair=True)
    with pytest.raises(
        DeepReadSchemaError, match=r"repair.*stage=map:research_question"
    ):
        DeepReadWorkflow(store, FakeEmbedder(), broken).generate(paper_id)
    assert len(broken.calls) == 2

    forged = ScriptedDeepReadLLM(forged_reduce=True)
    with pytest.raises(DeepReadSchemaError, match="未检索到"):
        DeepReadWorkflow(store, FakeEmbedder(), forged).generate(paper_id)
    store.close()


def test_deep_read_enforces_retrieval_and_context_budgets(tmp_path):
    store, paper_id = _store(tmp_path)
    llm = ScriptedDeepReadLLM()
    workflow = DeepReadWorkflow(
        store,
        FakeEmbedder(),
        llm,
        budget=DeepReadBudget(max_retrieval_calls=1),
    )
    with pytest.raises(DeepReadBudgetExceeded, match="retrieval_calls"):
        workflow.generate(paper_id)
    assert len(llm.calls) == 1
    store.close()


def test_deep_read_field_dedupes_duplicate_refs_instead_of_failing():
    """真实模型会对同一证据多次引用；schema 在解析前确定性去重。"""
    duplicated_id = "ev_" + "a" * 60
    field = DeepReadField.model_validate(
        {
            "text": "同一证据支持字段内的两个论断。",
            "evidence_refs": [
                {"evidence_id": duplicated_id, "quote": "第一次引用原文"},
                {"evidence_id": duplicated_id, "quote": "第二次引用原文"},
            ],
            "insufficient_evidence": False,
        }
    )
    assert len(field.evidence_refs) == 1
    assert field.evidence_refs[0].quote == "第一次引用原文"


def test_deep_read_field_prefers_valid_refs_over_contradictory_insufficient_flag():
    evidence_id = "ev_" + "c" * 60
    field = DeepReadField.model_validate(
        {
            "text": "该字段有明确证据支持。",
            "evidence_refs": [
                {"evidence_id": evidence_id, "quote": "verbatim evidence"}
            ],
            "insufficient_evidence": True,
        }
    )

    assert field.insufficient_evidence is False
    assert field.evidence_refs[0].evidence_id == evidence_id


def test_quote_recovery_restores_whitespace_drift_to_exact_substring():
    """模型转录的空白漂移恢复为真原文子串；非空白改动保持 fail closed。"""
    from pragent.research.deep_read import _locate_exact_quote

    text = (
        "We present the first investigation demonstrating\n"
        "   that imbalanced codebooks give rise to over-popular tokens."
    )
    located = _locate_exact_quote(text, "demonstrating that imbalanced")
    assert located == "demonstrating\n   that imbalanced"
    assert located in text
    # PDF 断词连字符：原文 "perfor-\nmance"，模型转写为 "performance"。
    hyphenated = "Overall, CRAB achieves perfor-\nmance comparable to MOR."
    assert _locate_exact_quote(hyphenated, "achieves performance comparable") == (
        "achieves perfor-\nmance comparable"
    )
    assert _locate_exact_quote(text, "demonstrating that imbalanceX") is None


def test_recover_field_quotes_updates_drifted_quotes():
    from types import SimpleNamespace

    from pragent.research.deep_read import _recover_field_quotes

    text = "alpha beta\ngamma delta"
    evidence_id = "ev_" + "b" * 60
    evidence = {evidence_id: SimpleNamespace(text=text)}
    value = DeepReadField.model_validate(
        {
            "text": "引用。",
            "evidence_refs": [
                {"evidence_id": evidence_id, "quote": "alpha beta gamma delta"}
            ],
        }
    )
    recovered = _recover_field_quotes(value, evidence)
    assert recovered is not value
    assert recovered.evidence_refs[0].quote == "alpha beta\ngamma delta"
    assert recovered.evidence_refs[0].quote in text
