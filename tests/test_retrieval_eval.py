import json
from pathlib import Path

import numpy as np
import pytest

from pragent.models import Chunk, Paper
from pragent.retrieval_eval import (
    AnswerReview,
    RelevantChunk,
    RetrievalCase,
    RetrievalEvalFormatError,
    load_retrieval_cases,
    run_retrieval_evaluation,
)
from pragent.store import Store


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class EvalEmbedder:
    model_name = "eval-model"

    def embed(self, texts, batch_size=32):
        vectors = {
            "keyword alpha": np.array([1.0, 0.0], dtype=np.float32),
            "semantic beta": np.array([0.0, 1.0], dtype=np.float32),
        }
        return np.stack([vectors.get(text, np.array([1.0, 1.0])) for text in texts])


def _paper(path, sha):
    return Paper(
        id=None,
        path=path,
        sha256=sha,
        title=path,
        authors=["A"],
        year=2026,
        page_count=1,
        has_text=True,
        indexed_at="2026-01-01T00:00:00",
    )


def _store(tmp_path):
    first_sha = "a" * 64
    second_sha = "b" * 64
    store = Store(tmp_path / "eval.db")
    store.upsert_paper(
        _paper("a.pdf", first_sha),
        [Chunk(None, 0, 0, 1, "keyword alpha", np.array([1.0, 0.0]))],
    )
    store.upsert_paper(
        _paper("b.pdf", second_sha),
        [Chunk(None, 0, 0, 1, "other words", np.array([0.0, 1.0]))],
    )
    store.upsert_paper(
        _paper("c.pdf", "c" * 64),
        [Chunk(None, 0, 0, 1, "third distractor", np.array([0.5, 0.5]))],
    )
    store.meta_set("embed_model", "eval-model")
    return store, first_sha, second_sha


def test_load_retrieval_cases_validates_version_ids_and_relevance(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "name": "fixture",
                "cases": [
                    {
                        "id": "q1",
                        "query": "问题",
                        "relevant": [
                            {
                                "paper_sha256": "a" * 64,
                                "chunk_seq": 0,
                                "evidence_excerpt": "证据",
                            }
                        ],
                        "tags": ["unit"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    name, cases = load_retrieval_cases(path)
    assert name == "fixture"
    assert cases[0].relevant[0].key == ("a" * 64, 0)

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["cases"].append(dict(payload["cases"][0]))
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RetrievalEvalFormatError, match="重复"):
        load_retrieval_cases(path)


def test_load_retrieval_cases_enforces_declared_case_count(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "name": "fixture",
                "expected_case_count": 2,
                "cases": [
                    {
                        "id": "q1",
                        "query": "问题",
                        "relevant": [
                            {
                                "paper_sha256": "a" * 64,
                                "chunk_seq": 0,
                                "evidence_excerpt": "证据",
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RetrievalEvalFormatError, match="预期 2"):
        load_retrieval_cases(path)


def test_committed_core_dataset_has_balanced_thirty_case_snapshot():
    name, cases = load_retrieval_cases(
        PROJECT_ROOT / "benchmarks" / "retrieval" / "core_v1.json"
    )

    assert name == "pra-core-30-v1"
    assert len(cases) == 30
    assert sum("zh" in case.tags for case in cases) == 15
    assert sum("en" in case.tags for case in cases) == 15
    assert {case.id.split("-", 1)[0] for case in cases} == {
        "crab",
        "baco",
        "recoatlas",
    }


def test_run_retrieval_evaluation_rejects_stale_relevance_labels(tmp_path):
    store, first_sha, _ = _store(tmp_path)
    cases = [
        RetrievalCase(
            id="q1",
            query="keyword alpha",
            relevant=(RelevantChunk(first_sha, 0, "不存在的摘录"),),
        )
    ]

    with pytest.raises(RetrievalEvalFormatError, match="与当前索引不一致"):
        run_retrieval_evaluation(store, EvalEmbedder(), "fixture", cases)
    store.close()


def test_run_retrieval_evaluation_rejects_unrelated_answer_evidence(tmp_path):
    store, first_sha, _ = _store(tmp_path)
    cases = [
        RetrievalCase(
            id="q1",
            query="keyword alpha",
            relevant=(RelevantChunk(first_sha, 0, "keyword alpha"),),
            answer_review=AnswerReview(
                answer="结论 [E:ev_not_relevant]。",
                allowed_evidence_ids=("ev_not_relevant",),
            ),
        )
    ]

    with pytest.raises(RetrievalEvalFormatError, match="非相关或已漂移"):
        run_retrieval_evaluation(store, EvalEmbedder(), "fixture", cases)
    store.close()


def test_run_retrieval_evaluation_reports_metrics_failures_and_citation_boundaries(
    tmp_path,
):
    store, first_sha, second_sha = _store(tmp_path)
    first_evidence = store.evidence_from_chunk(1).id
    second_evidence = store.evidence_from_chunk(2).id
    cases = [
        RetrievalCase(
            id="q1",
            query="keyword alpha",
            relevant=(RelevantChunk(first_sha, 0, "keyword alpha"),),
            answer_review=AnswerReview(
                answer=f"结论 [{first_evidence.replace('ev_', 'E:ev_')}]。",
                allowed_evidence_ids=(first_evidence,),
                human_support="supported",
            ),
        ),
        RetrievalCase(
            id="q2",
            query="semantic beta",
            relevant=(RelevantChunk(second_sha, 0, "other words"),),
            answer_review=AnswerReview(
                answer="伪造引用 [E:ev_not_allowed]。",
                allowed_evidence_ids=(second_evidence,),
                human_support="unsupported",
            ),
        ),
    ]

    report = run_retrieval_evaluation(
        store,
        EvalEmbedder(),
        "fixture",
        cases,
        modes=("bm25", "vector", "rrf"),
        top_k=1,
    )

    summaries = {summary.mode: summary for summary in report.mode_summaries}
    assert summaries["bm25"].recall_at_k == pytest.approx(0.5)
    assert summaries["vector"].recall_at_k == pytest.approx(1.0)
    assert summaries["vector"].mrr == pytest.approx(1.0)
    assert summaries["bm25"].failed_case_ids == ("q2",)
    assert report.citation_summary.machine_reviewed == 2
    assert report.citation_summary.validity_rate == pytest.approx(0.5)
    assert report.citation_summary.human_reviewed == 2
    assert report.citation_summary.support_rate == pytest.approx(0.5)
    assert report.to_dict()["dataset_name"] == "fixture"
    store.close()
