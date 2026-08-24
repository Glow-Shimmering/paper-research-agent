"""Offline Phase 3 smoke: project + Web document + hybrid search + Web pages."""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from pragent.ingestion.indexing import index_web_source
from pragent.search import hybrid_search
from pragent.storage import ResearchRepository
from pragent.store import Store
from pragent.webapp import create_app


class FixtureEmbedder:
    model_name = "phase3-smoke-fixture"

    def embed(self, texts, batch_size=32):
        rows = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            rows.append(np.frombuffer(digest[:32], dtype=np.uint8).astype(np.float32))
        return np.stack(rows)


class OfflineLLM:
    is_configured = False


class EmptyProvider:
    name = "fixture"

    def search(self, query, *, limit=10):
        return []

    def lookup(self, identifier):
        return None


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="pra-phase3-smoke-") as raw:
        root = Path(raw)
        store = Store(root / "smoke.db")
        repository = ResearchRepository(root / "smoke.db")
        project = repository.create_project("Phase 3 smoke")
        digest = hashlib.sha256(b"fixture html").hexdigest()
        source = repository.create_source(
            "url:https://example.org/report",
            "web",
            title="Fixture Web Report",
            canonical_url="https://example.org/report",
            content_sha256=digest,
            status="ready",
            snapshot_path=f"{digest}.html.gz",
            snapshot_sha256=digest,
            extracted_text="offline phase three hybrid evidence from a web document " * 30,
        )
        repository.add_source_identity(
            source.id, "url", "https://example.org/report", is_primary=True
        )
        repository.add_source_identity(source.id, "content_sha256", digest)
        repository.add_source_record(
            source.id, "fixture", "report", {"title": "Fixture Web Report"}
        )
        indexed = index_web_source(
            store,
            repository,
            source.id,
            FixtureEmbedder(),
            progress=lambda _message: None,
        )
        repository.add_project_source(project.id, indexed.source.id)
        hits = hybrid_search(
            store,
            FixtureEmbedder(),
            "offline phase three hybrid evidence",
            top=5,
        )
        assert hits and hits[0].source_kind == "web"

        app = create_app(
            store=store,
            research_repository=repository,
            embedder=FixtureEmbedder(),
            llm=OfflineLLM(),
            source_providers=[EmptyProvider()],
        )
        with TestClient(app, base_url="http://127.0.0.1") as client:
            assert client.get("/ui/discover").status_code == 200
            assert client.get("/ui/library").status_code == 200
            payload = client.get("/api/v1/sources").json()
            assert payload["total"] == 1
            assert "snapshot_path" not in str(payload)
            assert str(root) not in str(payload)
        repository.close()
        store.close()
    print("Phase 3 offline research smoke passed")


if __name__ == "__main__":
    main()
