"""可审计的离线检索评测：BM25 / vector / RRF、Recall@k、MRR 与引用边界。"""
from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .agent import Verifier
from .search import SearchMode, prepare_search_corpus, retrieval_search

RETRIEVAL_EVAL_SCHEMA_VERSION = 1
RETRIEVAL_MODES: tuple[SearchMode, ...] = ("bm25", "vector", "rrf")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_HUMAN_SUPPORT = frozenset({"not_reviewed", "supported", "mixed", "unsupported"})


class RetrievalEvalFormatError(ValueError):
    pass


@dataclass(frozen=True)
class RelevantChunk:
    paper_sha256: str
    chunk_seq: int
    evidence_excerpt: str
    note: str = ""

    @property
    def key(self) -> tuple[str, int]:
        return self.paper_sha256, self.chunk_seq


@dataclass(frozen=True)
class AnswerReview:
    answer: str
    allowed_evidence_ids: tuple[str, ...]
    human_support: str = "not_reviewed"
    note: str = ""


@dataclass(frozen=True)
class RetrievalCase:
    id: str
    query: str
    relevant: tuple[RelevantChunk, ...]
    tags: tuple[str, ...] = ()
    answer_review: AnswerReview | None = None


@dataclass(frozen=True)
class RetrievedChunk:
    rank: int
    evidence_id: str
    paper_sha256: str
    chunk_seq: int
    title: str
    page: int
    score: float

    @property
    def key(self) -> tuple[str, int]:
        return self.paper_sha256, self.chunk_seq


@dataclass(frozen=True)
class RetrievalCaseResult:
    case_id: str
    mode: str
    recall_at_k: float
    reciprocal_rank: float
    latency_ms: float
    relevant_ranks: tuple[int, ...]
    retrieved: tuple[RetrievedChunk, ...]
    error: str | None = None


@dataclass(frozen=True)
class ModeSummary:
    mode: str
    case_count: int
    recall_at_k: float
    mrr: float
    mean_latency_ms: float
    failed_case_ids: tuple[str, ...]


@dataclass(frozen=True)
class CitationSummary:
    machine_reviewed: int
    legal: int
    validity_rate: float | None
    human_reviewed: int
    supported: int
    support_rate: float | None


@dataclass(frozen=True)
class RetrievalEvaluationReport:
    dataset_name: str
    schema_version: int
    top_k: int
    mode_summaries: tuple[ModeSummary, ...]
    case_results: tuple[RetrievalCaseResult, ...]
    citation_summary: CitationSummary

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_retrieval_cases(path: str | Path) -> tuple[str, list[RetrievalCase]]:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RetrievalEvalFormatError(f"无法读取评测数据 {source}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise RetrievalEvalFormatError("评测数据必须是 JSON 对象")
    if payload.get("version") != RETRIEVAL_EVAL_SCHEMA_VERSION:
        raise RetrievalEvalFormatError(
            f"不支持的评测数据版本 {payload.get('version')!r}；"
            f"当前为 {RETRIEVAL_EVAL_SCHEMA_VERSION}"
        )
    name = _nonempty(payload.get("name"), "name")
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise RetrievalEvalFormatError("cases 必须是非空数组")
    expected_case_count = payload.get("expected_case_count")
    if expected_case_count is not None:
        if (
            not isinstance(expected_case_count, int)
            or isinstance(expected_case_count, bool)
            or expected_case_count <= 0
        ):
            raise RetrievalEvalFormatError("expected_case_count 必须是正整数")
        if len(raw_cases) != expected_case_count:
            raise RetrievalEvalFormatError(
                f"cases 数量为 {len(raw_cases)}，预期 {expected_case_count}"
            )
    cases: list[RetrievalCase] = []
    seen: set[str] = set()
    for raw in raw_cases:
        case = _parse_case(raw)
        if case.id in seen:
            raise RetrievalEvalFormatError(f"case id 重复：{case.id}")
        seen.add(case.id)
        cases.append(case)
    return name, cases


def run_retrieval_evaluation(
    store,
    embedder,
    dataset_name: str,
    cases: Iterable[RetrievalCase],
    *,
    modes: Iterable[SearchMode] = RETRIEVAL_MODES,
    top_k: int = 5,
) -> RetrievalEvaluationReport:
    if top_k <= 0:
        raise ValueError("top_k 必须大于 0")
    selected_modes = tuple(modes)
    if not selected_modes or any(mode not in RETRIEVAL_MODES for mode in selected_modes):
        raise ValueError("modes 只能包含 bm25、vector、rrf")
    case_list = list(cases)
    if not case_list:
        raise ValueError("cases 不能为空")

    _validate_relevance_against_store(store, case_list)
    prepare_search_corpus(store)
    if any(mode in {"vector", "rrf"} for mode in selected_modes):
        embedder.embed([case_list[0].query])
    results: list[RetrievalCaseResult] = []
    for case in case_list:
        relevant_keys = {item.key for item in case.relevant}
        for mode in selected_modes:
            started = time.perf_counter()
            try:
                hits = retrieval_search(
                    store,
                    embedder,
                    case.query,
                    mode=mode,
                    top=top_k,
                )
                retrieved = tuple(
                    _retrieved_chunk(store, hit, rank)
                    for rank, hit in enumerate(hits, start=1)
                )
                ranks = tuple(item.rank for item in retrieved if item.key in relevant_keys)
                recall = len({item.key for item in retrieved} & relevant_keys) / len(
                    relevant_keys
                )
                reciprocal_rank = 1.0 / min(ranks) if ranks else 0.0
                error = None
            except Exception as exc:
                retrieved = ()
                ranks = ()
                recall = 0.0
                reciprocal_rank = 0.0
                error = f"{type(exc).__name__}: {exc}"
            latency_ms = (time.perf_counter() - started) * 1000.0
            results.append(
                RetrievalCaseResult(
                    case_id=case.id,
                    mode=mode,
                    recall_at_k=recall,
                    reciprocal_rank=reciprocal_rank,
                    latency_ms=latency_ms,
                    relevant_ranks=ranks,
                    retrieved=retrieved,
                    error=error,
                )
            )

    summaries = tuple(
        _summarize_mode(mode, [result for result in results if result.mode == mode])
        for mode in selected_modes
    )
    return RetrievalEvaluationReport(
        dataset_name=dataset_name,
        schema_version=RETRIEVAL_EVAL_SCHEMA_VERSION,
        top_k=top_k,
        mode_summaries=summaries,
        case_results=tuple(results),
        citation_summary=_citation_summary(case_list),
    )


def _validate_relevance_against_store(store, cases: list[RetrievalCase]) -> None:
    chunks_by_key: dict[tuple[str, int], tuple[str, str]] = {}
    for paper in store.iter_papers():
        for chunk in store.paper_chunks(paper.id):
            evidence_id = store.evidence_from_chunk(chunk.id).id
            chunks_by_key[(paper.sha256, chunk.seq)] = (chunk.text, evidence_id)

    for case in cases:
        relevant_evidence_ids: set[str] = set()
        for relevant in case.relevant:
            indexed = chunks_by_key.get(relevant.key)
            if indexed is None:
                raise RetrievalEvalFormatError(
                    f"{case.id} 引用了当前索引不存在的 chunk："
                    f"{relevant.paper_sha256}:{relevant.chunk_seq}"
                )
            text, evidence_id = indexed
            if relevant.evidence_excerpt not in text:
                raise RetrievalEvalFormatError(
                    f"{case.id} 的 evidence_excerpt 与当前索引不一致："
                    f"{relevant.paper_sha256}:{relevant.chunk_seq}"
                )
            relevant_evidence_ids.add(evidence_id)
        if case.answer_review is not None:
            unknown = set(case.answer_review.allowed_evidence_ids) - relevant_evidence_ids
            if unknown:
                raise RetrievalEvalFormatError(
                    f"{case.id}.answer_review 引用了非相关或已漂移的 evidence id："
                    f"{', '.join(sorted(unknown))}"
                )


def _retrieved_chunk(store, hit, rank: int) -> RetrievedChunk:
    evidence = store.evidence_from_chunk(hit.chunk_id)
    return RetrievedChunk(
        rank=rank,
        evidence_id=evidence.id,
        paper_sha256=evidence.paper_sha256,
        chunk_seq=evidence.chunk_seq,
        title=evidence.title,
        page=evidence.page,
        score=float(hit.score),
    )


def _summarize_mode(mode: str, results: list[RetrievalCaseResult]) -> ModeSummary:
    count = len(results)
    return ModeSummary(
        mode=mode,
        case_count=count,
        recall_at_k=sum(result.recall_at_k for result in results) / count,
        mrr=sum(result.reciprocal_rank for result in results) / count,
        mean_latency_ms=sum(result.latency_ms for result in results) / count,
        failed_case_ids=tuple(
            result.case_id
            for result in results
            if result.error is not None or result.recall_at_k < 1.0
        ),
    )


def _citation_summary(cases: list[RetrievalCase]) -> CitationSummary:
    reviews = [case.answer_review for case in cases if case.answer_review is not None]
    legal = 0
    human_reviews = []
    for review in reviews:
        assert review is not None
        if Verifier().verify_evidence(
            review.answer, review.allowed_evidence_ids
        ).ok:
            legal += 1
        if review.human_support != "not_reviewed":
            human_reviews.append(review)
    supported = sum(review.human_support == "supported" for review in human_reviews)
    return CitationSummary(
        machine_reviewed=len(reviews),
        legal=legal,
        validity_rate=(legal / len(reviews)) if reviews else None,
        human_reviewed=len(human_reviews),
        supported=supported,
        support_rate=(supported / len(human_reviews)) if human_reviews else None,
    )


def _parse_case(raw: Any) -> RetrievalCase:
    if not isinstance(raw, Mapping):
        raise RetrievalEvalFormatError("每个 case 必须是对象")
    case_id = _nonempty(raw.get("id"), "case.id")
    query = _nonempty(raw.get("query"), f"{case_id}.query")
    raw_relevant = raw.get("relevant")
    if not isinstance(raw_relevant, list) or not raw_relevant:
        raise RetrievalEvalFormatError(f"{case_id}.relevant 必须是非空数组")
    relevant = tuple(_parse_relevant(case_id, item) for item in raw_relevant)
    if len({item.key for item in relevant}) != len(relevant):
        raise RetrievalEvalFormatError(f"{case_id}.relevant 存在重复 chunk")
    tags_raw = raw.get("tags", [])
    if not isinstance(tags_raw, list) or any(not isinstance(tag, str) for tag in tags_raw):
        raise RetrievalEvalFormatError(f"{case_id}.tags 必须是字符串数组")
    review = _parse_answer_review(case_id, raw.get("answer_review"))
    return RetrievalCase(case_id, query, relevant, tuple(tags_raw), review)


def _parse_relevant(case_id: str, raw: Any) -> RelevantChunk:
    if not isinstance(raw, Mapping):
        raise RetrievalEvalFormatError(f"{case_id}.relevant 项必须是对象")
    sha = str(raw.get("paper_sha256") or "").lower()
    if not _SHA256_RE.fullmatch(sha):
        raise RetrievalEvalFormatError(f"{case_id}.paper_sha256 必须是 64 位小写十六进制")
    seq = raw.get("chunk_seq")
    if not isinstance(seq, int) or isinstance(seq, bool) or seq < 0:
        raise RetrievalEvalFormatError(f"{case_id}.chunk_seq 必须是非负整数")
    excerpt = _nonempty(raw.get("evidence_excerpt"), f"{case_id}.evidence_excerpt")
    note = str(raw.get("note") or "").strip()
    return RelevantChunk(sha, seq, excerpt, note)


def _parse_answer_review(case_id: str, raw: Any) -> AnswerReview | None:
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise RetrievalEvalFormatError(f"{case_id}.answer_review 必须是对象")
    answer = _nonempty(raw.get("answer"), f"{case_id}.answer_review.answer")
    allowed = raw.get("allowed_evidence_ids")
    if (
        not isinstance(allowed, list)
        or not allowed
        or any(not str(item).strip() for item in allowed)
    ):
        raise RetrievalEvalFormatError(
            f"{case_id}.answer_review.allowed_evidence_ids 必须是非空字符串数组"
        )
    support = str(raw.get("human_support") or "not_reviewed")
    if support not in _HUMAN_SUPPORT:
        raise RetrievalEvalFormatError(f"{case_id}.human_support 不合法")
    return AnswerReview(
        answer=answer,
        allowed_evidence_ids=tuple(str(item) for item in allowed),
        human_support=support,
        note=str(raw.get("note") or "").strip(),
    )


def _nonempty(value: Any, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise RetrievalEvalFormatError(f"{field} 不能为空")
    return text
