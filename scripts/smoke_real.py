"""End-to-end smoke test using the configured real embedding model.

The first run may download the embedding model. No LLM or arXiv API is called.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import fitz

from pragent import config
from pragent.embeddings import Embedder
from pragent.indexer import index_library
from pragent.search import hybrid_search
from pragent.store import Store


def _make_pdf(path: Path) -> None:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(
        (72, 72),
        "Attention retrieval systems combine lexical and semantic evidence. " * 8,
    )
    doc.save(path)
    doc.close()


def main() -> None:
    temp_root = Path(__file__).resolve().parent.parent / ".pytest-tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="real-smoke-", dir=temp_root) as raw:
        root = Path(raw)
        papers = root / "papers"
        papers.mkdir()
        _make_pdf(papers / "smoke.pdf")
        store = Store(root / "library.db")
        try:
            embedder = Embedder(config.EMBED_MODEL)
            result = index_library(store, papers, embedder, progress=lambda _: None)
            hits = hybrid_search(store, embedder, "semantic retrieval", top=1)
            if result["added"] != 1 or not hits or hits[0].title != "smoke":
                raise SystemExit(f"real smoke failed: index={result}, hits={hits}")
            print(
                f"real smoke passed: model={config.EMBED_MODEL}, "
                f"dim={embedder.dim}, score={hits[0].score:.6f}"
            )
        finally:
            store.close()


if __name__ == "__main__":
    main()
