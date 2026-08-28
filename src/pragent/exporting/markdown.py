"""Deterministic evidence-first Markdown artifact rendering."""

from __future__ import annotations

from pragent.research import DEEP_READ_FIELD_LABELS

from .common import citation_output, parse_artifact_content, review_drafts
from .models import FrozenArtifactExport


def render_markdown(snapshot: FrozenArtifactExport) -> bytes:
    content = parse_artifact_content(snapshot)
    citations = citation_output(snapshot, content)
    labels = iter(citations.citations)
    lines = [
        f"# {snapshot.artifact.title or _content_title(content)}",
        "",
        f"> Project: {snapshot.project.title}",
        (
            f"> Artifact: `{snapshot.artifact.artifact_type}` / revision "
            f"{snapshot.revision.revision_number}"
        ),
        f"> Citation style: `{snapshot.citation_style}`",
        f"> Freshness: {'stale' if snapshot.freshness.stale else 'current'}",
        "",
    ]
    artifact_type = snapshot.artifact.artifact_type
    if artifact_type == "deep_read":
        _render_deep_read(lines, content, labels)
    elif artifact_type == "comparison":
        _render_comparison(lines, snapshot, content, labels)
    elif artifact_type == "review_outline":
        _render_outline(lines, snapshot, content, labels)
    elif artifact_type == "review_section":
        _render_section(lines, content, labels)
    _render_references(lines, citations.bibliography)
    _render_evidence_appendix(lines, snapshot)
    return ("\n".join(lines).rstrip() + "\n").encode("utf-8")


def _render_deep_read(lines, card, labels) -> None:
    for name, field in card.ordered_fields():
        lines.extend((f"## {DEEP_READ_FIELD_LABELS[name]}", ""))
        if field.insufficient_evidence:
            lines.extend(("证据不足。", ""))
        else:
            lines.extend((_with_citation(field.text, next(labels)), ""))


def _render_comparison(lines, snapshot, matrix, labels) -> None:
    sources = snapshot.source_by_id()
    lines.extend(("## 比较矩阵", ""))
    header = ["维度", *(_escape_table(sources[item].title) for item in matrix.source_ids)]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join("---" for _item in header) + " |")
    cells = {(item.source_id, item.dimension_key): item for item in matrix.cells}
    citation_labels = {
        (item.source_id, item.dimension_key): next(labels) for item in matrix.cells
    }
    for dimension in matrix.dimensions:
        row = [_escape_table(dimension.label)]
        for source_id in matrix.source_ids:
            cell = cells[(source_id, dimension.key)]
            text = "证据不足" if cell.insufficient_evidence else cell.summary
            row.append(
                _escape_table(
                    _with_citation(
                        text, citation_labels[(source_id, dimension.key)]
                    )
                )
            )
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")


def _render_outline(lines, snapshot, outline, labels) -> None:
    lines.extend(("## 研究问题", ""))
    for question in outline.research_questions:
        lines.append(f"- {question.question}")
    lines.append("")
    drafts = review_drafts(snapshot)
    for section in outline.sections:
        lines.extend((f"## {section.title}", "", section.objective, ""))
        draft = drafts.get(section.key)
        claims = draft.claims if draft is not None else section.planned_claims
        if draft is None:
            lines.extend(("_本节尚未生成草稿；以下为提纲计划。_", ""))
        for claim in claims:
            text = "证据不足" if claim.insufficient_evidence else claim.text
            lines.append(f"- {_with_citation(text, next(labels))}")
        lines.append("")


def _render_section(lines, section, labels) -> None:
    lines.extend((f"## {section.section_title}", ""))
    for claim in section.claims:
        text = "证据不足" if claim.insufficient_evidence else claim.text
        lines.extend((_with_citation(text, next(labels)), ""))


def _render_references(lines, bibliography) -> None:
    lines.extend(("## 参考文献", ""))
    lines.extend(bibliography or ("无。",))
    lines.append("")


def _render_evidence_appendix(lines, snapshot) -> None:
    lines.extend(("## Evidence appendix", ""))
    if not snapshot.evidence:
        lines.extend(("无。", ""))
        return
    for item in snapshot.evidence:
        evidence = item.evidence
        page = f", p. {evidence.page}" if evidence.page > 0 else ""
        lines.extend(
            (
                f"### `{evidence.id}`",
                "",
                f"{evidence.title}{page}; field `{item.link.field_path}`",
                "",
                "> " + evidence.text.replace("\n", "\n> "),
                "",
            )
        )


def _content_title(content) -> str:
    return getattr(content, "title", getattr(content, "section_title", "研究导出"))


def _with_citation(text: str, citation: str) -> str:
    return f"{text} {citation}".rstrip()


def _escape_table(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", "<br>")
