from dataclasses import replace
from pathlib import Path

import pytest

from pragent.models import ResearchSource
from pragent.research import (
    STYLE_REGISTRY,
    CitationStyleError,
    render_bibliography,
    render_citation_cluster,
    render_citation_document,
    source_to_csl_json,
)


def _source(**changes):
    values = {
        "id": "source_demo",
        "canonical_key": "doi:10.1234/demo",
        "source_kind": "paper",
        "title": "Evidence First Research Agents",
        "authors": ("Alice Chen", "Bob Smith"),
        "year": 2024,
        "doi": "10.1234/demo",
        "arxiv_id": None,
        "canonical_url": "https://doi.org/10.1234/demo",
        "content_sha256": None,
        "indexed_paper_id": 1,
        "status": "ready",
        "metadata": {
            "container-title": ["Journal of Research Systems"],
            "volume": "12",
            "issue": "3",
            "page": "101-120",
            "publisher": "Example Press",
            "language": "en",
        },
        "locator": {},
        "snapshot_path": None,
        "snapshot_sha256": None,
        "extracted_text": None,
        "fetched_at": None,
        "version": 1,
        "created_at": "2024-01-01T00:00:00+00:00",
        "updated_at": "2024-01-01T00:00:00+00:00",
    }
    values.update(changes)
    return ResearchSource(**values)


def test_source_metadata_normalizes_to_csl_json_without_inventing_fields():
    item = source_to_csl_json(_source())

    assert item == {
        "id": "source_demo",
        "type": "article-journal",
        "title": "Evidence First Research Agents",
        "author": [
            {"family": "Chen", "given": "Alice"},
            {"family": "Smith", "given": "Bob"},
        ],
        "issued": {"date-parts": [[2024]]},
        "DOI": "10.1234/demo",
        "URL": "https://doi.org/10.1234/demo",
        "container-title": "Journal of Research Systems",
        "volume": "12",
        "issue": "3",
        "page": "101-120",
        "publisher": "Example Press",
        "language": "en",
    }
    web = source_to_csl_json(
        _source(
            id="source_web",
            source_kind="web",
            authors=("张伟",),
            year=None,
            doi=None,
            canonical_url="https://example.test/report",
            fetched_at="2026-08-28T09:30:00+00:00",
            metadata={},
        )
    )
    assert web["type"] == "webpage"
    assert web["author"] == [{"literal": "张伟"}]
    assert web["accessed"] == {"date-parts": [[2026, 8, 28]]}
    assert "issued" not in web and "DOI" not in web


@pytest.mark.parametrize(
    ("style", "citation", "bibliography"),
    [
        (
            "gb-t-7714-2015-numeric",
            "[1]",
            "[1]CHEN A, SMITH B. Evidence First Research Agents[J/OL]. Journal of Research Systems, 2024, 12(3): 101-120. https://doi.org/10.1234/demo. DOI:10.1234/demo.",
        ),
        (
            "apa-7",
            "(Chen & Smith, 2024)",
            "Chen, A., & Smith, B. (2024). Evidence First Research Agents. Journal of Research Systems, 12(3), 101–120. https://doi.org/10.1234/demo",
        ),
        (
            "ieee",
            "[1]",
            "[1]A. Chen and B. Smith, “Evidence First Research Agents”, Journal of Research Systems, vol. 12, no. 3, pp. 101–120, 2024, doi: 10.1234/demo.",
        ),
        (
            "chicago-author-date",
            "(Chen and Smith 2024)",
            "Chen, A., and B. Smith. 2024. “Evidence First Research Agents”. Journal of Research Systems 12 (3): 101–20. https://doi.org/10.1234/demo.",
        ),
        (
            "mla",
            "(Chen and Smith, “Evidence First Research Agents”)",
            "Chen, A., and B. Smith. “Evidence First Research Agents”. Journal of Research Systems, vol. 12, no. 3, 2024, pp. 101–20, https://doi.org/10.1234/demo.",
        ),
    ],
)
def test_five_bundled_styles_match_golden_output(style, citation, bibliography):
    source = _source()
    assert render_citation_cluster([source], style) == citation
    assert render_bibliography([source], style) == (bibliography,)


def test_style_registry_is_complete_licensed_and_fails_closed():
    assert [item.key for item in STYLE_REGISTRY] == [
        "gb-t-7714-2015-numeric",
        "apa-7",
        "ieee",
        "chicago-author-date",
        "mla",
    ]
    style_dir = Path(__file__).parents[1] / "src" / "pragent" / "styles"
    for spec in STYLE_REGISTRY:
        text = style_dir.joinpath(spec.filename).read_text(encoding="utf-8")
        assert spec.csl_id in text
        assert "creativecommons.org/licenses/by-sa/3.0" in text
    attribution = style_dir.joinpath("ATTRIBUTION.md").read_text(encoding="utf-8")
    assert "2a4430b7cadae7cc88012537c5ceaed76d1d9938" in attribution
    with pytest.raises(CitationStyleError, match="不支持"):
        render_bibliography([_source()], "invented-style")


def test_document_renderer_keeps_numeric_clusters_and_bibliography_aligned():
    first = _source()
    second = replace(
        first,
        id="source_second",
        canonical_key="doi:10.1234/second",
        title="Second Evidence Source",
        authors=("Carol Jones",),
        doi="10.1234/second",
        canonical_url="https://doi.org/10.1234/second",
    )

    rendered = render_citation_document(
        [first, second],
        [(second.id,), (first.id, second.id)],
        "ieee",
    )

    assert rendered.citations == ("[1]", "[1], [2]")
    assert rendered.bibliography[0].startswith("[1]C. Jones")
    assert rendered.bibliography[1].startswith("[2]A. Chen")
