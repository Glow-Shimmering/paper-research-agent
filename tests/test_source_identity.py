import itertools

import pytest

from pragent.sources import (
    NormalizedSource,
    canonicalize_url,
    deduplicate_sources,
    normalize_arxiv_id,
    normalize_content_sha256,
    normalize_doi,
    source_identities,
)


def record(provider, record_id, **kwargs):
    return NormalizedSource(
        provider=provider,
        provider_record_id=record_id,
        metadata={"provider": provider, "record_id": record_id},
        **kwargs,
    )


def test_normalize_doi_arxiv_url_and_content_hash():
    assert normalize_doi("https://doi.org/10.1000/ABC.Def") == "10.1000/abc.def"
    assert normalize_doi("doi:10.48550/arXiv.2402.11651") == "10.48550/arxiv.2402.11651"
    assert normalize_doi("not-a-doi") is None

    assert normalize_arxiv_id("arXiv:2402.11651v3") == "2402.11651"
    assert normalize_arxiv_id("https://arxiv.org/pdf/2402.11651v2.pdf") == "2402.11651"
    assert normalize_arxiv_id("hep-th/9901001v4") == "hep-th/9901001"
    assert normalize_arxiv_id("2402.bad") is None

    digest = "A" * 64
    assert normalize_content_sha256(digest) == digest.lower()
    assert normalize_content_sha256("abc") is None
    assert canonicalize_url(
        "HTTPS://Example.COM:443/reports/../paper/?utm_source=x&b=2&a=1#section"
    ) == "https://example.com/paper?a=1&b=2"
    assert canonicalize_url("https://例子.测试/研究") == (
        "https://xn--fsqu00a.xn--0zwm56d/%E7%A0%94%E7%A9%B6"
    )
    with pytest.raises(ValueError, match="credentials"):
        canonicalize_url("https://user:pass@example.com/paper")


def test_identity_priority_is_doi_then_arxiv_url_and_hash():
    item = record(
        "arxiv",
        "2402.11651",
        doi="10.1000/Test",
        arxiv_id="2402.11651v2",
        canonical_url="https://example.org/paper",
        content_sha256="f" * 64,
    )
    assert source_identities(item) == (
        ("doi", "10.1000/test"),
        ("arxiv", "2402.11651"),
        ("url", "https://example.org/paper"),
        ("content_sha256", "f" * 64),
    )


def test_deduplicate_sources_is_transitive_deterministic_and_keeps_provenance():
    crossref = record(
        "crossref",
        "10.1000/X",
        title="Canonical Crossref Title",
        authors=("Alice",),
        year=2024,
        doi="10.1000/X",
    )
    bridge = record(
        "semantic_scholar",
        "Corpus-1",
        title="Semantic title",
        doi="https://doi.org/10.1000/x",
        arxiv_id="arXiv:2401.00001v2",
    )
    arxiv = record(
        "arxiv",
        "2401.00001",
        title="Arxiv title",
        arxiv_id="2401.00001v1",
        canonical_url="https://arxiv.org/abs/2401.00001v1",
        pdf_url="https://arxiv.org/pdf/2401.00001",
    )

    expected = deduplicate_sources([crossref, bridge, arxiv])
    assert len(expected) == 1
    merged = expected[0]
    assert merged.canonical_key == "doi:10.1000/x"
    assert merged.source.title == "Canonical Crossref Title"
    assert merged.source.doi == "10.1000/x"
    assert merged.source.arxiv_id == "2401.00001"
    assert merged.providers == ("arxiv", "crossref", "semantic_scholar")
    assert merged.duplicate_count == 2
    assert [(item.provider, item.record_id) for item in merged.provenance] == [
        ("arxiv", "2401.00001"),
        ("crossref", "10.1000/X"),
        ("semantic_scholar", "Corpus-1"),
    ]

    for permutation in itertools.permutations([crossref, bridge, arxiv]):
        assert deduplicate_sources(permutation) == expected


def test_similar_title_is_not_identity_and_record_fallback_is_stable():
    first = record("provider_a", "1", title="Same title")
    second = record("provider_b", "2", title="Same title")
    duplicate_first = record("provider_a", "1", title="Same title, refreshed")

    merged = deduplicate_sources([second, duplicate_first, first])

    assert len(merged) == 2
    first_group = next(item for item in merged if item.source.provider == "provider_a")
    assert first_group.duplicate_count == 0
    assert first_group.canonical_key.startswith("record_sha256:")
