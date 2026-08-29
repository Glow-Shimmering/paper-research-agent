"""Live provider smoke：手动验证 arXiv / Semantic Scholar / Crossref 实时可用性。

需要用户显式运行（roadmap：live 结果单独记录，不能拿 fixture 冒充实时可用性）：

    python scripts/smoke_live_providers.py --query "retrieval augmented generation"
    python scripts/smoke_live_providers.py --providers arxiv crossref --json report.json

行为：
- 每个 provider 独立执行一次有界搜索并记录：日期、query、HTTP 语义结果、
  返回条数、耗时、限流/缺字段提示；单 provider 失败不影响其他 provider；
- arXiv 免费无需 key；Semantic Scholar / Crossref 读取可选的
  PRA_SEMANTIC_SCHOLAR_API_KEY / PRA_CROSSREF_EMAIL 环境变量；
- 输出脱敏 JSON：只含题录计数与稳定 id（DOI/arXiv id），不含摘要正文。
退出码：0 = 至少一个 provider 成功；1 = 全部失败。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from pragent.sources import (
    ArxivAdapter,
    CrossrefAdapter,
    SemanticScholarAdapter,
)


def _run_provider(name, provider, query: str, limit: int) -> dict:
    started = time.monotonic()
    try:
        records = provider.search(query, limit=limit)
    except Exception as exc:
        return {
            "provider": name,
            "ok": False,
            "error_code": str(getattr(exc, "code", exc.__class__.__name__))[:80],
            "error_message": str(exc)[:200],
            "elapsed_seconds": round(time.monotonic() - started, 2),
        }
    return {
        "provider": name,
        "ok": True,
        "count": len(records),
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "sample_identifiers": [
            {
                "doi": record.doi,
                "arxiv_id": record.arxiv_id,
                "title_chars": len(record.title or ""),
            }
            for record in records[:3]
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", default="retrieval augmented generation")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument(
        "--providers",
        nargs="*",
        default=["arxiv", "semantic_scholar", "crossref"],
        choices=["arxiv", "semantic_scholar", "crossref"],
    )
    parser.add_argument("--json", type=Path, help="把报告写入该 JSON 文件")
    args = parser.parse_args()

    report: dict = {
        "kind": "pragent-live-provider-smoke",
        "query": args.query,
        "limit": args.limit,
        "ran_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "results": [],
    }

    builders = {
        "arxiv": lambda: ArxivAdapter(),
        "semantic_scholar": lambda: SemanticScholarAdapter(
            api_key=os.getenv("PRA_SEMANTIC_SCHOLAR_API_KEY", "")
        ),
        "crossref": lambda: CrossrefAdapter(
            contact_email=os.getenv("PRA_CROSSREF_EMAIL", "")
        ),
    }
    for name in args.providers:
        print(f"搜索 {name} …", file=sys.stderr)
        report["results"].append(
            _run_provider(name, builders[name](), args.query, args.limit)
        )

    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(payload + "\n", encoding="utf-8")
        print(f"报告已写入：{args.json}")
    print(payload)

    ok = any(item["ok"] for item in report["results"])
    if not ok:
        print("全部 provider 失败；请检查网络或 provider 状态。", file=sys.stderr)
    else:
        print("提示：请把本次日期/query/限流与缺字段情况记入评估文档。")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
