# 检索评测基线：`pra-core-30-v1`

更新时间：2026-08-27。本页记录评测脚手架的首个真实三论文快照；它是下一阶段检索调整的 **before** 基线，不包含任何针对结果的调参。

## 数据与审计边界

- 数据文件：`benchmarks/retrieval/core_v1.json`，schema version 1，声明并校验 30 题。
- 每篇论文 10 题；15 个英文查询、15 个中文跨语言查询。
- 每题固定 `paper_sha256`、`chunk_seq` 和必须能在当前分块中精确找到的 `evidence_excerpt`。
- 运行前会用 `(paper_sha256, chunk_seq)` 反查 SQLite；分块不存在、摘录漂移或回答白名单引用不属于相关分块时，评测直接失败，不产生看似有效的指标。
- 当前每题只有一个相关分块，因此此处的 Recall@5 等价于“30 题中相关分块进入前五的比例”。这不是多相关文档基准。

固定论文快照：

| 论文 | SHA-256 | 分块数 |
|---|---|---:|
| CRAB | `028da916f1a7d990cec1d59566a3f1e36330a54dcdcb4b8589548ef5cdcc7d76` | 34 |
| BACO | `32e8124f463f57f95036a496527f1408fc4a7b79c10138fc1e47d6d265a8e057` | 106 |
| RecoAtlas | `76a411df9394d7ea28e74a0ab94b41f398200b42077281e31ba6dbf7b61db5e4` | 119 |

## 复现

先按 `week-01-baseline.md` 将三篇 PDF 放入被 Git 忽略的 `sample_papers/`，再在 PowerShell 中使用独立数据目录：

```powershell
$env:PRA_DATA_DIR = "D:\path\to\pra-eval-data"
.venv\Scripts\pra.exe index sample_papers --force
.venv\Scripts\python.exe scripts\evaluate_retrieval.py `
  benchmarks\retrieval\core_v1.json `
  --db "$env:PRA_DATA_DIR\library.db" `
  --output "$env:PRA_DATA_DIR\retrieval-report.json"
```

脚本在计时前预构建检索语料缓存，并对 embedding 做一次预热。JSON 报告包含每个 mode 的汇总、每题命中排名、返回 evidence、延迟和异常；运行失败与前五未命中都进入 `failed_case_ids`。

## 首次真实结果

环境：Windows，本地 CPU embedding `BAAI/bge-small-zh-v1.5`，259 个分块，`top_k=5`，单次预热后运行。延迟只是这台机器的一次本地测量，不应外推为稳定性能承诺。

| 模式 | Recall@5 | MRR | 平均查询延迟 | 未命中 |
|---|---:|---:|---:|---:|
| BM25 | 0.6000 | 0.5067 | 1.536 ms | 12/30 |
| vector | 0.4333 | 0.3428 | 2.732 ms | 17/30 |
| RRF | 0.6667 | 0.5122 | 3.895 ms | 10/30 |

失败题目：

- BM25：`crab-02-popular-definition`、`crab-03-two-stage-method`、`crab-04-token-groups`、`crab-06-hierarchical-regularizer`、`crab-08-protocol-metrics`、`crab-10-splitting-ratio`、`baco-04-objective-hardness`、`baco-06-label-update`、`baco-08-complexity`、`baco-10-speedups`、`recoatlas-04-three-views`、`recoatlas-10-query-mixture`。
- vector：`crab-03-two-stage-method`、`crab-05-regularized-kmeans`、`crab-06-hierarchical-regularizer`、`crab-07-training-objective`、`crab-09-implementation`、`crab-10-splitting-ratio`、`baco-02-louvain-limit`、`baco-04-objective-hardness`、`baco-07-secondary-clusters`、`baco-08-complexity`、`baco-09-evaluation-protocol`、`baco-10-speedups`、`recoatlas-01-online-ab`、`recoatlas-03-output-constraints`、`recoatlas-04-three-views`、`recoatlas-06-episode-count`、`recoatlas-07-tool-contribution`。
- RRF：`crab-03-two-stage-method`、`crab-04-token-groups`、`crab-07-training-objective`、`crab-10-splitting-ratio`、`baco-04-objective-hardness`、`baco-08-complexity`、`baco-10-speedups`、`recoatlas-03-output-constraints`、`recoatlas-04-three-views`、`recoatlas-10-query-mixture`。

## 引用与人工复核

数据中有 5 条示例回答进入引用检查，5/5 只使用了其相关分块对应的当前 evidence ID，引用合法率为 1.0000。这里验证的是引用格式、白名单范围和快照身份，不证明自然语言答案被证据语义蕴含。

所有示例回答的 `human_support` 当前均为 `not_reviewed`，所以人工已复核数为 0，人工证据支持率为 `null`（N/A），不能写成 0% 或 100%。只有人实际逐条检查后，才可改为 `supported`、`mixed` 或 `unsupported`。本次评测不调用 LLM，token 用量为 0。

## 下一步边界

下一阶段只根据本报告的 paired query 失败明细提出一个可解释的检索调整，重新运行同一数据与快照，并独立提交 before/after 指标和逐题 delta。不得改问题或相关标签来美化调优结果。
