"""Run the versioned retrieval benchmark against an existing PRAgent index."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pragent.embeddings import Embedder
from pragent.retrieval_eval import load_retrieval_cases, run_retrieval_evaluation
from pragent.store import Store


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    name, cases = load_retrieval_cases(args.dataset)
    store = Store(args.db)
    try:
        model_name = args.model or store.meta_get("embed_model")
        if not model_name:
            raise SystemExit("索引没有 embed_model；请先建立索引或显式传 --model")
        report = run_retrieval_evaluation(
            store,
            Embedder(model_name),
            name,
            cases,
            top_k=args.top_k,
        )
    finally:
        store.close()

    payload = json.dumps(report.to_dict(), ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)


if __name__ == "__main__":
    main()
