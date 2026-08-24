"""Offline Phase 4 smoke: real text PDF -> bounded Deep Read artifact."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import fitz
import numpy as np

from pragent.indexer import index_library
from pragent.research import DEEP_READ_FIELD_ORDER, DeepReadArtifactService, DeepReadWorkflow
from pragent.storage import ResearchRepository
from pragent.store import Store


class FixtureEmbedder:
    model_name = "phase4-smoke-fixture"

    def embed(self, texts, batch_size=32):
        rows = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            rows.append(np.frombuffer(digest[:32], dtype=np.uint8).astype(np.float32))
        return np.stack(rows)


class ScriptedLLM:
    is_configured = True
    model = "phase4-scripted-llm"

    def __init__(self):
        self.calls = 0

    def chat_with_metadata(self, system, user):
        self.calls += 1
        if "精读助手" in system:
            payload = json.loads(user)
            evidence = payload["evidence"]
            if evidence:
                quote = evidence[0]["text"][:80]
                content = {
                    "text": f"{payload['label']}的离线总结",
                    "evidence_refs": [
                        {"evidence_id": evidence[0]["evidence_id"], "quote": quote}
                    ],
                    "insufficient_evidence": False,
                }
            else:
                content = {
                    "text": "证据不足",
                    "evidence_refs": [],
                    "insufficient_evidence": True,
                }
        else:
            content = json.loads(user)
        return {
            "content": json.dumps(content, ensure_ascii=False),
            "metadata": {
                "usage": {"total_tokens": 10},
                "finish_reason": "stop",
                "response_id": f"smoke-{self.calls}",
            },
        }


def _write_pdf(path: Path) -> None:
    document = fitz.open()
    page = document.new_page()
    text = (
        "This study asks how evidence-grounded retrieval improves research agents. "
        "The method combines field-specific retrieval with structured generation. "
        "Experiments on Dataset A report higher citation accuracy. "
        "Limitations include text-only PDF extraction, and future work studies tables. "
    ) * 12
    page.insert_textbox(fitz.Rect(50, 50, 545, 790), text, fontsize=9)
    document.save(path)
    document.close()


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="pra-phase4-smoke-") as raw:
        root = Path(raw)
        papers = root / "papers"
        papers.mkdir()
        _write_pdf(papers / "real-text.pdf")
        store = Store(root / "research.db")
        embedder = FixtureEmbedder()
        indexed = index_library(store, papers, embedder, progress=lambda _message: None)
        assert indexed["added"] == 1
        _, indexed_papers = store.list_papers(None, 10, 0)
        paper = indexed_papers[0]

        repository = ResearchRepository(root / "research.db")
        project = repository.create_project("Phase 4 offline smoke")
        source = repository.ensure_source_for_paper(paper.id)
        repository.add_project_source(project.id, source.id)
        workflow = DeepReadWorkflow(store, embedder, ScriptedLLM())
        saved = DeepReadArtifactService(repository).generate_and_save(
            project.id, source.id, workflow
        )

        assert saved.revision.revision_number == 1
        assert repository.artifact_freshness(saved.artifact.id).stale is False
        assert {
            link.field_path
            for link in repository.list_artifact_evidence(saved.revision.id)
        } == {f"$.{name}" for name in DEEP_READ_FIELD_ORDER}
        repository.close()
        store.close()
    print("Phase 4 offline real-text PDF Deep Read smoke passed")


if __name__ == "__main__":
    main()
