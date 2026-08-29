# PRAgent 评估与质量证据

本文是 PRAgent 的评估口径：哪些质量声明由确定性测试支撑、哪些必须由真实
provider/LLM 的 live 证据支撑、以及人工质量检查使用的 rubric。原则只有一条：

> 脚本化测试只能证明产品合同（schema、预算、引用范围、恢复边界），永远不能
> 被用来宣称模型质量、提示注入抵抗能力或语义蕴含验证。

## 1. 确定性产品场景（离线）

### 四条核心用户旅程

`tests/test_product_journeys.py` 把 roadmap「Manual product journeys」的四条旅程
做成无网络、确定性的端到端回归（FakeEmbedder + 脚本化 LLM + fixture
provider/fetcher/downloader）：

| 旅程 | 场景测试 | 覆盖合同 |
|---|---|---|
| 单篇精读 | `test_journey_single_paper_deep_read_edit_export_and_restart` | 九栏 schema、证据抽屉（evidence/页码/quote）、人工编辑新 revision、Markdown/DOCX 导出、服务重启后项目/artifact/历史恢复 |
| 多篇比较/综述 | `test_journey_three_paper_compare_review_style_switch_and_export` | 3 来源比较矩阵、提纲、章节草稿、GB/T→APA 样式切换、CSV round-trip、JSON schema、DOCX |
| 发现与入库 | `test_journey_discovery_dedupe_download_webfetch_hybrid_search` | 多 provider 聚合、DOI dedupe、PDF 下载索引、网页快照索引、同一次 hybrid search 命中 PDF+Web、加入项目 |
| 新鲜度与恢复 | `test_journey_staleness_job_restart_and_agent_pending` | 重新索引→stale→旧版本可读→重生成、任务重启 interrupted→幂等重排、Agent 待确认跨重启确认 |

运行：

```bash
python -m pytest tests/test_product_journeys.py -q
```

这些场景验证的是状态、边界与恢复行为；字段文本来自脚本化 LLM，不代表真实
DeepSeek 的生成质量。

### 其他确定性资产

- 37 个 Agent 状态机/引用合同场景：`tests/scenarios/core_v1.json` + `pragent.agent_eval`；
- 检索评测数据集与脚本：`scripts/evaluate_retrieval.py`（见
  [检索评测基线](ownership/retrieval-evaluation.md)）；
- CSL 引用 golden tests、导出 determinism、SSRF 合同、CSRF/API key 合同等
  分布于 `tests/`（完整清单见 `docs/architecture.md` 测试边界一节）。

## 2. 人工质量 rubric（用于 live 结果检查）

对 live DeepSeek 产出的每张精读卡、每节综述，逐项打分（0=不成立，1=部分成立，
2=完全成立）。**任一关键项为 0 即视为该产物不通过**，与自动化测试无关。

### 精读卡（每栏目）

| 检查项 | 判定问题 |
|---|---|
| 证据支持 | 栏目结论能否从所引 evidence 原文直接读出？（引用身份合法 ≠ 语义蕴含，必须人工读） |
| Quote 保真 | quote 是否为原文逐字子串、页码是否对得上（点击证据抽屉核对） |
| 忠实度 | 是否夸大（把"略有提升"写成"显著突破"）、是否把相关工作写成自己的贡献 |
| 完整性 | 数据集/实验栏目是否给出版本/规模等可核对细节，而不是空泛描述 |
| 局限性 | 局限栏是否如实来自论文，而不是模型客套（"样本较少"式敷衍记 1 分） |
| 语言 | 中文总结 + 英文证据保留原文 |

### 比较矩阵与综述

| 检查项 | 判定问题 |
|---|---|
| Cell 归属 | 每个 cell 摘要是否只来自该来源的精读卡 evidence，没有跨来源张冠李戴 |
| 逐 claim 映射 | 综述每条 claim 的 source/evidence token 是否指向正确来源与原文 |
| 证据不足 | 证据不足时是否显式标注（而不是静默编造）；`insufficient_evidence` 是否被滥用为偷懒 |
| 提纲对齐 | 章节是否覆盖研究问题，planned claims 是否被正文实际使用 |

### 引用样式

五种内置样式（GB/T 7714-2015、APA 7、IEEE、Chicago author-date、MLA）各取
一条真实来源人工核对 citation 与 bibliography 格式；golden tests 只证明
processor 集成稳定，不证明格式学的正确性。

### 系统级指标

| 指标 | 来源 | 说明 |
|---|---|---|
| Recall@5 / MRR | `scripts/evaluate_retrieval.py` | 检索质量随人工标注集演进 |
| 引用合法率 | Agent 场景 evaluation + live 抽样 | 确定性部分：非法 evidence ID 必须被拒 |
| 人工证据支持率 | live 抽样按上述 rubric 统计 | `支持数 / 抽样 claim 数` |
| 延迟 / token | live 证据记录中的 elapsed/usage | 按 step 分开记录 |

## 3. Live 证据的记录要求

### DeepSeek 工作流（需用户授权 + 密钥）

```bash
export PRA_LLM_API_KEY=...   # 密钥只进环境变量；不写入任何文件
python scripts/smoke_live_deepseek.py --pdf a.pdf --pdf b.pdf --pdf c.pdf --json live-deepseek.json
```

脚本输出脱敏 JSON（模型、usage、finish_reason、耗时、revision、错误码），
要求的最低覆盖：**2 张单篇卡、1 个三篇比较、1 个综述 section**。真实结果不可
预先写死；provider 未返回 usage/finish_reason 时保存 `null/unknown`。
之后按第 2 节 rubric 人工检查并把结论（分数 + 问题描述）记录在此文件或
独立的评估笔记中——**live 结果与 scripted 结果分开记录，不得混写**。

### Provider 实时可用性

```bash
python scripts/smoke_live_providers.py --query "retrieval augmented generation" --json live-providers.json
```

记录字段：日期、query、provider、结果数、HTTP 语义错误码、限流/缺字段。
arXiv 无需 key；Semantic Scholar/Crossref 的 key/邮箱只经环境变量读取。

## 4. 隐私/安全审查

```bash
python scripts/security_review.py --wheel dist
```

确定性静态检查：跟踪文件与 wheel 中的可用 key 字面量、`.env` 跟踪状态、
本机绝对路径、wheel 打包内容（不得含 `.env`/snapshot/PDF）、`|safe` 使用清单。
脚本内置检测器自检（对已知样例必须报警、对空值/占位符不误报）。

roadmap 其余安全边界的既有覆盖映射（均有合同测试，勿重复实现）：

| 检查 | 覆盖 |
|---|---|
| 公开 API/UI 不返回主机路径/snapshot locator/raw metadata/抽取全文 | `tests/test_research_web.py`、`tests/test_discovery_web.py`、`tests/test_dashboard_web.py` |
| 原始 HTML 不注入应用 DOM；snapshot 只存 gzip 文本 | `tests/test_snapshots.py`、`tests/test_web_ingestion.py`、`docs/source-security.md` |
| SSRF（私网/重定向/MIME/大小/超时/DNS pinning） | `tests/test_safe_fetch.py`、`tests/test_websearch.py`、`tests/test_download.py` |
| 远程监听要求 API key + TLS；CSRF；body limit | `tests/test_webapp.py`、`tests/test_security.py` |
| UI 明示 DeepSeek 接收范围与隐私边界 | `tests/test_dashboard_web.py`（帮助页/页脚合同） |
| citation scope ≠ 语义蕴含的声明 | 本文第 2 节 + 帮助页文案 |

## 5. 数据备份恢复演练

```bash
python scripts/backup_restore_drill.py [--keep]
```

在临时目录构建完整数据目录（索引、项目、精读卡、笔记、网页快照、任务），
文件级备份后校验 SHA-256 清单，再在全新空目录恢复并逐项验证
（项目/问题/来源/revision/evidence links/笔记/任务/snapshot hash/hybrid search）。
对应 roadmap 验收项「备份 `~/.pragent` 后可在空目录恢复……；旧 `~/.pagent`
始终不变」（演练全程不触碰真实数据目录）。

## 6. 发布验收（Step 29 口径）

```bash
python -m pip install -c requirements-dev.lock -e ".[dev]"
python -m pip check
python -m pytest -q
python scripts/check_tmp_space.py
python -m pip wheel . --no-deps --wheel-dir dist
python scripts/check_wheel.py dist
python scripts/security_review.py --wheel dist
python scripts/backup_restore_drill.py
```

最后在干净 venv 中安装 wheel 并验收 `pra --version` 与 `pra serve`
（空数据目录启动、`/ui/` 可访问、静态资源来自 wheel）；命令序列见
[README 构建与测试](../README.md#构建与测试)。
