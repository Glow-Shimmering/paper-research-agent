# 第一次评测驱动检索调整

更新时间：2026-08-27。本阶段冻结 `pra-core-30-v1`、三篇论文 SHA 和 259 个分块，只修改 RRF 融合权重。完整 before 基线与复现命令见[检索评测基线](retrieval-evaluation.md)。

## 诊断

等权 RRF 的 BM25 与 vector 单路 Recall@5 分别为 0.6000 和 0.4333。`crab-07-training-objective` 与 `recoatlas-03-output-constraints` 等题在 BM25 前五命中、vector 未命中，却被等权融合挤出前五；说明当前本地向量模型的弱排名有时会稀释强词法信号。

这不意味着应该删除向量路由。15 个中文跨语言问题仍需要语义召回，而且 BM25 无命中时，向量路由应继续独立贡献候选。因此调整只把 BM25 的 reciprocal-rank contribution 从 `1.0 / (60 + rank)` 改为 `1.5 / (60 + rank)`；vector 保持 `1.0 / (60 + rank)`。

在同一 30 题开发集上探索了少量固定候选权重。`1.25` 与 `1.5` 都把 Recall@5 提高到 0.7000；选择 `1.5` 是因为其 MRR 略高，且 paired ranks 没有退化。由于选择和报告使用同一数据，这个 after 结果是 in-sample development evidence，不是未见数据上的泛化证明。

## Paired 结果

同一数据库、embedding 模型、查询顺序和 `top_k=5`：

| 指标 | 等权 RRF | BM25 1.5 / vector 1.0 | Delta |
|---|---:|---:|---:|
| Recall@5 | 0.6667 | 0.7000 | +0.0333 |
| MRR | 0.5122 | 0.5594 | +0.0472 |
| 平均查询延迟 | 3.895 ms | 3.534 ms | -0.361 ms |
| 未命中 | 10/30 | 9/30 | -1 |

延迟差来自两次单机运行噪声；权重乘法没有增加检索阶段，不能据此宣称稳定提速。

逐题变化：

| Query | Before rank | After rank | 变化 |
|---|---:|---:|---|
| `crab-05-regularized-kmeans` | 3 | 2 | 改善 |
| `baco-07-secondary-clusters` | 2 | 1 | 改善 |
| `recoatlas-04-three-views` | 未进入前五 | 4 | 新增命中 |
| `recoatlas-07-tool-contribution` | 2 | 1 | 改善 |

其余 26 题前五命中状态和相关分块排名不变；没有 paired regression。调整后的 RRF 仍未命中 9 题：`crab-03-two-stage-method`、`crab-04-token-groups`、`crab-07-training-objective`、`crab-10-splitting-ratio`、`baco-04-objective-hardness`、`baco-08-complexity`、`baco-10-speedups`、`recoatlas-03-output-constraints`、`recoatlas-10-query-mixture`。

## 结论与下一道门

本次调整足以证明“指标发现问题 → 做窄修改 → 用相同 query 做 paired audit”的闭环，但不能证明 1.5 是通用最优权重。进一步调参前应新增未参与本次选择的 holdout 问题；同时由真人复核数据中 5 条 answer/evidence 对，将人工支持率从 N/A 转为可报告指标。继续在这 30 题上搜索权重会增加过拟合风险。
