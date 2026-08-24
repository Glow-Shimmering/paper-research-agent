# Paper Research Agent（PRAgent）产品化改进计划

> **范围纠正：上一版 Java v0.2 计划是误批准，不执行。**  
> `pagent-java/` 保持当前状态，专门用于用户尚未开始的“14 天 × 4 小时”学习路线；本计划改为从 Python `paper-agent/` 派生一个独立产品目录，面向用户真实论文写作与调研工作流。

## Context

用户希望的不是立即强化学习仓库，而是：

- 保留 `pagent-java/` 作为亲自学习 LangChain4j/Java Agent 的练习项目；
- 在新目录中改进现有 Pagent，使其成为更好的研究 Agent 和更易用的个人论文产品；
- 日常能够用于论文发现、入库、深读、跨论文比较、证据管理、笔记与写作材料整理；
- 新项目应复用 `paper-agent/` 已验证能力，而不是从头重写。

当前 Python Pagent 已具备较强底座：增量 PDF 索引、BM25+本地向量+RRF、普通问答、受控 Agent、稳定 evidence ID/freshness、arXiv 搜索/下载、笔记、CLI/TUI/Web 和 225 个测试。现有 Web 只有“检索/问答/Agent/论文库”四个标签；笔记和 evidence 能力主要埋在工具/TUI 中，尚未形成围绕真实研究任务的产品工作区。

## Decision

- **不修改 `paper-agent/` 或 `pagent-java/`。**
- 从 `paper-agent/` 派生独立 Python 仓库和目录 `paper-research-agent/`；产品简称 **PRAgent**，启动命令为 **`pra`**，原仓库保留为稳定基线和回归参照。
- 优化重点从“继续堆 Agent 工具”转为“研究工作流产品化”：项目工作区、可视化证据/笔记、结构化调研产出、可恢复运行、真实可用性与质量评估。
- `pagent-java/` 继续只服务于用户尚未开始的 14 天学习计划；本计划不包含任何 Java 改动。

## Confirmed product requirements

- 工作流优先级：**单篇精读 > 多篇比较/文献综述 > 论文搜索下载管理**；
- UI：**Web-first**，CLI/TUI 只保留兼容入口；
- 导出：Markdown、DOCX/Word、CSV、JSON；
- 来源：本地 PDF、arXiv、Semantic Scholar、Crossref、普通网页/博客/技术报告；
- 规模：100–1000 篇个人论文库；
- 模型：DeepSeek 云端，允许发送命中的论文片段；
- 语言：中文 UI/总结，英文论文 evidence 保留原文；
- 精读栏目：研究问题、相关工作、核心方法、创新点、数据集与实验、主要结果、局限性、未来工作、关键原文证据；
- 图表边界：首版仅处理可抽取文本，不做图片/表格视觉理解或 OCR；
- 综述：同时支持结构化对比/提纲与带 evidence 的章节草稿；
- 引用：通过 CSL style registry 支持主流格式；
- 网页：抽取正文加入本地检索，并保存可追溯原始快照；
- 前端：FastAPI + HTMX 轻量 Web；
- 命名：独立目录/仓库 `paper-research-agent`，简称 `pragent`，命令 `pra`；`pagent-java/` 保持不动。

## Discovered reuse and gaps

- `Store` 已有 `papers/chunks/evidence/agent_runs/agent_events`，且有 revision、CAS、WAL 和 freshness；可直接复用，不需要另建检索数据库。
- 当前没有 research project/workspace、project-paper membership、结构化研究问题、研究报告或数据库笔记模型；`save_note` 只是写 notes 目录中的单个文件。
- Web API 目前只有 status/papers/search/ask/websearch/reindex 与 agent run；Web 页面只有检索、问答、Agent、论文库四个 tab，evidence/notes/deep-reading tools 没有产品化界面。
- 已有 `search_within_paper/get_paper_outline/read_pages/read_chunk_context/pin_evidence/get_evidence/list_evidence`，因此单篇精读应编排和展示这些能力，而不是再造底层工具。
- 已有 arXiv 搜索与安全 PDF 下载；`websearch.py` 和 `download.py` 都是 arXiv 专用。Semantic Scholar/Crossref/普通网页需要 provider adapter、统一 source identity、去重和可追溯快照。
- Web Agent session 是进程内状态（最多 64 个），刷新或重启会丢消息；run/event 虽持久化，但不能恢复研究工作区。
- 当前依赖没有 DOCX 生成或网页正文抽取库；Markdown/CSV/JSON 可用标准库，DOCX 和 HTML extraction 需要新增有界依赖。
- 对 100–1000 篇个人库，SQLite WAL + 当前本地 embedding 路线可继续使用；本期不因规模引入向量数据库。
- 为让网页正文进入同一检索/evidence 链路，新产品应把索引对象提升为通用 `Document`（PDF/Web snapshot），而不是另建一套孤立的 web search。新运行目录使用 `~/.pragent`；旧 `~/.pagent` 不原地迁移，只通过显式 `pra import-pagent` 复制导入。
- 学术来源统一为 provider adapters；DOI、arXiv ID、canonical URL、内容 SHA 按优先级形成 identity，Crossref/Semantic Scholar/arXiv 的同一论文合并为一项，同时保留 provider provenance/raw metadata。
- 普通网页原始 HTML 使用 content-addressed gzip snapshot 文件保存，SQLite 只存 URI/hash/路径/抽取正文/元数据；网络 fetch 必须逐跳防 SSRF、限类型/大小/超时/重定向并阻止私网地址。
- Web UI 采用 FastAPI + Jinja2 + 本地打包 HTMX，SSE/少量交互保留小型原生 JS；不增加 Node/SPA 构建链。
- 正文抽取优先复用 Trafilatura；DOCX 复用 `python-docx`；CSV/JSON/Markdown 使用标准库；引用格式通过 CSL processor + bundled styles 扩展，首批至少验证 GB/T 7714、APA、IEEE，并允许后续增加其他 CSL style。
- 深度生成任务需要 SQLite-backed bounded job queue，避免 HTTP 请求阻塞数分钟；DeepSeek 输出必须经过 Pydantic schema 验证，并保存 model/usage/finish reason/source revision。

## Approach

### 1. 独立派生而不是修改学习仓库

- 从 `paper-agent/` 保留 Git 历史派生到 `/Users/glow/Documents/pi/pi_src/paper-research-agent/`，原仓库和 `pagent-java/` 均不修改。
- 新 distribution 名为 `paper-research-agent`，Python package 名为 `pragent`，产品简称 **PRAgent**，唯一主命令为 `pra`。
- 默认数据目录改为 `~/.pragent`；提供显式、默认 dry-run 的 `pra import-pagent --source ~/.pagent`，复制旧库内容后再升级，绝不原地修改 `~/.pagent`。
- 先保留并迁移现有 225 个测试，再开始产品功能；已有 CLI/TUI/JSON API 可兼容，但新功能以 Web 为主。

### 2. 在现有检索底座上增加研究工作区

SQLite 继续作为 100–1000 篇个人库的 source of truth。新增真实版本化 migration，而不是当前“建表后直接提高 schema_version”的弱升级方式。核心模型：

- `research_projects`：主题、描述、默认语言 `zh-CN`、引用样式、状态；
- `research_questions`：项目内研究问题与排序；
- `research_sources`：统一文献/网页实体，保存 canonical identity、题录、状态和关联的 indexed paper；
- `source_records`：arXiv/Semantic Scholar/Crossref/Web 的 provider provenance 与原始元数据；
- `project_sources`：用户明确选择进入项目的来源；
- `research_artifacts` + `artifact_revisions`：精读卡、比较矩阵、综述提纲/草稿及人工编辑历史；
- `artifact_evidence`：artifact 字段/段落到稳定 evidence ID 的映射；
- `research_notes`：项目、来源或 evidence 级 Markdown 笔记；
- `research_jobs`：可恢复的后台生成/抓取/导出任务。

现有 `papers/chunks/evidence` 继续承担本地全文索引。为减少无价值的大重命名，本版不改表名；给 indexed paper 增加 source kind/canonical URI/locator metadata，让 PDF 与 Web snapshot 走同一 BM25/向量/evidence 链路。公开 API/UI 统一称 `source/document`，不泄露兼容表命名。

### 3. 统一论文发现、去重和安全入库

定义统一 provider contract：`search()`、`lookup()`、`normalize()`，实现：

- arXiv：复用现有 `websearch.py`、`download.py` 的限速和原子 PDF 校验；
- Semantic Scholar：Graph API，字段白名单、1 req/s 默认节流、可选 API key；
- Crossref：REST API，使用 contact email/User-Agent、缓存和 429 backoff；
- Web：用户显式 URL 导入，Trafilatura 抽取正文与元数据。

身份合并顺序：normalized DOI → normalized arXiv ID → canonical URL → content SHA-256。Provider 记录不丢失；冲突字段显示来源，由确定规则选择主值，不能让 LLM 猜测去重。

网页 fetch 逐跳执行 SSRF 防护：只允许 HTTP(S)、禁止 URL credentials、解析并拒绝 loopback/private/link-local/multicast/reserved IP、每次重定向重新校验、限制重定向数/超时/响应大小/MIME。原始 HTML 以 SHA-256 命名并 gzip 保存到 `~/.pragent/snapshots/`；数据库保存 hash、最终 URL、抓取时间、抽取正文和 snapshot 相对路径。UI 永不直接渲染原始 HTML。

### 4. 以有界工作流实现精读，而不是一次超长 Prompt

“生成精读卡”是显式后台 job：

1. 读取 outline，按字段执行单篇检索并去重 evidence；
2. 对长论文分批 map，得到带 evidence ID 和精确原文 quote 的候选；
3. reduce 为 Pydantic `DeepReadCard`；
4. 验证 evidence ID 属于当前来源、quote 是 snapshot 原文子串、字段完整；失败只 repair 一次；
5. 原子保存 artifact revision、模型、usage、finish reason、prompt/schema version 和 source fingerprint。

固定栏目按用户要求：研究问题、相关工作、核心方法、创新点、数据集与实验、主要结果、局限性、未来工作、关键原文证据。默认中文总结，英文证据保持原文。支持只重新生成一个字段；用户编辑形成新 revision，不覆盖模型原稿。源内容变化后 artifact 标记 stale，但旧版本保留。

### 5. 复用精读卡生成比较矩阵与文献综述

- 用户在 project 中明确选择 2–20 个来源；缺失精读卡时先排队生成，不让 Agent在整库中任意选论文。
- 比较矩阵默认以精读栏目为维度，并允许用户添加自定义维度；每个 cell 保存摘要、source ID、evidence IDs 和缺证据状态。
- 文献综述先生成结构化提纲，再按 section 生成带证据的章节草稿；跨论文 claim 必须关联至少一个当前 project source/evidence，否则标记“证据不足”。
- 内部使用结构化 citation tokens，而不是让模型直接拼最终 APA/IEEE 文本；导出阶段再渲染引用样式。
- 图像、表格理解和 OCR 本版不做；精读只使用可抽取文本。

### 6. 使用 SQLite-backed bounded job queue

- 深读、比较、综述、网页抓取和导出都进入持久 `research_jobs`；固定少量 worker，按 job type 设 timeout/token/source 数量预算。
- 状态：queued/running/succeeded/failed/cancel_requested/cancelled/interrupted；CAS claim 防止重复执行。
- 启动时将遗留 running 标为 interrupted，仅对声明 idempotent 的 job 显式重排；artifact 只在完整校验成功后提交。
- Web 通过 HTMX polling/SSE 查看进度和错误；取消只能保证工作流在阶段边界停止，除非底层 SDK 支持，否则不声称能中断正在进行的 DeepSeek HTTP 请求。

### 7. Web-first 研究工作台

使用 FastAPI + Jinja2 + 本地打包 HTMX；保留少量原生 JS 处理 SSE，不引入 Node 构建链。页面：

- Dashboard：项目、最近来源、运行中任务；
- Discover：多 provider 合并搜索、去重结果、抓取/下载/加入项目；
- Library：100–1000 条分页筛选、来源状态、重新索引；
- Project Workspace：Overview/Questions、Sources、Deep Reads、Compare、Review、Evidence & Notes；
- Agent：限定在当前 project 和已选来源内，展示工具/evidence/run，写入/联网仍走确认。

现有 JSON API 保留兼容，新能力放 `/api/v1/...`；HTMX 使用独立 `/ui/...` fragment routes。所有 destructive form 使用 CSRF token；默认只监听 loopback，远程模式继续要求 API key + TLS。

### 8. 多格式导出与 CSL 引用

- Markdown：综述正文、引用、参考文献和 evidence appendix；
- DOCX：标题层级、段落、比较表、静态格式化引文、参考文献和证据附录；
- CSV：来源目录和比较矩阵；
- JSON：完整 artifact schema、revision、provenance、evidence 和模型元数据。

引用统一转换为 CSL-JSON，通过 style registry 渲染。首批内置并做 golden tests：GB/T 7714-2015、APA 7、IEEE、Chicago author-date、MLA；新增样式只需放入受许可的 CSL 文件。若 processor 不支持某个 CSL 扩展，必须明确失败或使用经过测试的 formatter，不能静默生成错误格式。

### 9. 先修影响产品正确性的现有缺陷

- Web SSE disconnect：worker 未结束前不释放同 session 排他权；增加 cancel event，迟到事件丢弃并记录终态。
- Tool timeout：不再只声明 metadata；I/O handler 接收 deadline/cancel token，网络超时使用剩余预算，长循环检查取消。禁止用“future 超时但副作用线程继续跑”伪装安全 timeout。
- Web Agent session/transcript/pending confirmation 持久化并绑定 project；刷新页面可恢复，服务重启后 pending action 只能按冻结参数继续/取消。
- 普通问答/Agent/研究 artifact 继续执行 citation scope/freshness 校验；来自网页的正文同样视为 untrusted prompt data。

## Files to modify

所有实现只发生在新目录 `paper-research-agent/`；以下源路径均以该目录为根。创建新目录时从 `paper-agent/` 保留历史派生，再执行包名迁移。

### Existing files carried forward and changed

- `pyproject.toml`：distribution、`pra` entry point、新依赖、package data；
- `.env.example`、`README.md`、`docs/architecture.md`：改为 PRAgent 配置与产品说明；
- `src/paper_agent/**` → `src/pragent/**`：保留 Git rename history；
- `src/pragent/config.py`：`PRA_` 配置、`~/.pragent`、provider/job/snapshot limits；
- `src/pragent/models.py`：source/project/artifact/job/session domain models；
- `src/pragent/store.py`：保留索引/evidence 方法并接入真正 schema migrations；大段新 repository 逻辑必须拆包；
- `src/pragent/indexer.py`、`search.py`、`answer.py`：支持 web snapshot locator/source filtering/project scope；
- `src/pragent/llm.py`：JSON schema 调用、usage/finish metadata、deadline；
- `src/pragent/agent.py`、`chat.py`、`tool_protocol.py`、`tools.py`：project scope、持久 session/pending action、真实 deadline/cancel；
- `src/pragent/webapp.py`、`agent_api.py`：保留 app factory/security middleware，路由拆分；
- `src/pragent/web/*`：旧静态页迁移为 Jinja/HTMX 工作台；
- `tests/**`：更新 package/CLI 名，并保持所有旧合同测试。

### New critical modules

- `src/pragent/storage/migrations.py`
- `src/pragent/storage/research_repository.py`
- `src/pragent/storage/job_repository.py`
- `src/pragent/sources/base.py`
- `src/pragent/sources/identity.py`
- `src/pragent/sources/arxiv.py`
- `src/pragent/sources/semantic_scholar.py`
- `src/pragent/sources/crossref.py`
- `src/pragent/sources/web.py`
- `src/pragent/ingestion/safe_fetch.py`
- `src/pragent/ingestion/snapshots.py`
- `src/pragent/ingestion/html_extract.py`
- `src/pragent/research/schemas.py`
- `src/pragent/research/deep_read.py`
- `src/pragent/research/compare.py`
- `src/pragent/research/review.py`
- `src/pragent/research/citations.py`
- `src/pragent/jobs/queue.py`
- `src/pragent/jobs/worker.py`
- `src/pragent/exporting/markdown.py`
- `src/pragent/exporting/docx.py`
- `src/pragent/exporting/tabular.py`
- `src/pragent/exporting/json_export.py`
- `src/pragent/web/routes/{projects,discovery,library,artifacts,exports}.py`
- `src/pragent/web/templates/**`
- `src/pragent/web/static/{htmx.min.js,app.css,app.js}`
- `src/pragent/styles/*.csl` 与对应 license/attribution；
- `scripts/smoke_research.py`
- `docs/product-workflows.md`
- `docs/data-model.md`
- `docs/source-security.md`
- `docs/evaluation.md`

### New/updated tests

- `tests/test_migrations.py`
- `tests/test_import_pagent.py`
- `tests/test_research_store.py`
- `tests/test_source_identity.py`
- `tests/test_source_providers.py`
- `tests/test_safe_fetch.py`
- `tests/test_snapshots.py`
- `tests/test_jobs.py`
- `tests/test_deep_read.py`
- `tests/test_compare.py`
- `tests/test_review.py`
- `tests/test_exports.py`
- `tests/test_research_web.py`
- existing SSE/tool/session tests extended for disconnect, persistence and timeout behavior.

## Reuse

直接迁移并扩展现有经过测试的实现：

- 增量 PDF 索引、revision/CAS 与事务提交：`paper-agent/src/paper_agent/indexer.py`、`store.py`；
- 分块、fastembed、本地向量、BM25/RRF 和一致 snapshot cache：`chunking.py`、`embeddings.py`、`bm25.py`、`search.py`；
- 稳定 evidence ID、pin/get/list 和动态 freshness：`models.py:Evidence`、`store.py` evidence methods；
- 数字/evidence 引用 scope 验证和一次 repair：`agent.py`、`answer.py`、`chat.py`；
- 显式 FSM、预算、ToolSpec effects、冻结参数确认票据：`agent.py`、`tool_protocol.py`、`tools.py`；
- arXiv 限速搜索、PDF 上限/EOF/PyMuPDF 校验与原子替换：`websearch.py`、`download.py`；
- OpenAI-compatible DeepSeek sync/stream/tool call 及真实 usage metadata：`llm.py`；
- FastAPI app factory、loopback/API-key/origin/body-size security：`webapp.py`、`security.py`；
- run/event 审计和 37 个 deterministic Agent scenarios：`store.py`、`agent_eval.py`、`tests/scenarios/core_v1.json`；
- 225 个现有测试作为派生仓库的回归底线。

外部成熟能力只用于明确边界：Trafilatura（网页正文抽取）、Jinja2+HTMX（服务端 UI）、python-docx（DOCX）、CSL processor/styles（引用格式）。不自研 HTML readability、Word OOXML 或引用样式语言。

## Steps

### Phase 1 — 安全派生与兼容基线

- [x] Step 1：从 `paper-agent/` 派生 `paper-research-agent/`，保留提交历史但移除旧 remote；确认 `paper-agent/` 与 `pagent-java/` 均无改动。
- [x] Step 2：使用 Git rename 将 package 改为 `pragent`，更新 distribution/config/static branding 和所有测试 import；注册唯一命令 `pra`，版本从 `0.1.0` 开始。
- [x] Step 3：在改功能前跑通旧的 pytest、wheel/package-data、CLI/TUI/Web smoke，记录派生基线。

**Gate：** `pra --version/status/serve` 可用；旧 225 个测试语义全部保留；旧两个仓库 Git 状态不变。

### Phase 2 — 数据模型、迁移和项目工作区

- [x] Step 4：建立顺序 schema migrations 与 backup/rollback 边界；新增 project/source/artifact/revision/evidence-link/note/job/session/pending-action tables。
- [x] Step 5：实现 research/source/job repositories、CAS/version 检查、分页查询和 artifact freshness；避免把新 SQL 继续堆入 1100 行 `store.py`。
- [x] Step 6：实现 `pra import-pagent`：默认 dry-run、校验旧 DB/schema/文件、复制到临时目录、迁移验证后原子落位；目标已存在时 fail closed。
- [x] Step 7：完成最小 Web vertical slice：创建项目、编辑研究问题、从现有本地论文库选择来源、刷新后恢复。

**Gate：** 新库、从 v0.7 导入库、失败回滚三种路径均测试；项目/问题/来源在服务重启后存在。

### Phase 3 — 多来源发现、去重与全文入库

- [x] Step 8：定义 provider/normalized-source contract 和 deterministic identity merge；将现有 arXiv 实现迁入 adapter。
- [x] Step 9：实现 Semantic Scholar/Crossref adapters、fixture cache、rate limit/backoff、可选认证和 provider provenance。
- [x] Step 10：实现 SSRF-safe Web fetch、gzip content-addressed snapshots、Trafilatura extraction 与 metadata normalization。
- [ ] Step 11：让网页正文和下载 PDF 进入现有 chunk/embed/search/evidence 管线；source 关联 indexed paper，API/UI 不暴露 snapshot/主机路径。
- [ ] Step 12：实现 Discover/Library HTMX 页面：多 provider 聚合、dedupe badges、加入项目、下载/抓取/index 状态和错误恢复。

**Gate：** 同一 DOI/arXiv work 的多 provider 结果只显示一个 canonical source；恶意/私网/过大/错误 MIME URL 被拒绝；本地 PDF 与网页正文可被同一次 hybrid search 命中。

### Phase 4 — 持久后台任务与单篇精读

- [ ] Step 13：实现 SQLite-backed bounded job queue、CAS claim、progress、cancel/interrupted/restart 与幂等重排。
- [ ] Step 14：实现 DeepReadCard Pydantic schema、field-specific retrieval、map/reduce、一次 JSON repair、token/context/tool budgets。
- [ ] Step 15：验证 evidence scope、原文 quote、source fingerprint；原子保存 artifact+revision+evidence links+真实模型 metadata。
- [ ] Step 16：实现 Deep Read UI：生成进度、九个固定栏目、证据抽屉、单字段重生成、人工编辑和版本历史。

**Gate：** 对真实文本 PDF 完成精读卡；每个非空事实字段可展开到当前 evidence/页码/原文；篡造 ID/quote 被拒；源更新后旧卡显示 stale 而不丢失。

### Phase 5 — 跨论文比较与综述

- [ ] Step 17：实现 project-scoped comparison workflow，限制 2–20 个已选来源，复用精读卡并支持自定义维度。
- [ ] Step 18：实现比较矩阵 UI/编辑/version；每个 cell 保存 evidence 或明确 `insufficient_evidence`。
- [ ] Step 19：实现 review outline workflow，基于研究问题、选择来源和比较结果规划章节。
- [ ] Step 20：实现 section draft workflow 与编辑/version；claim 使用结构化 source/evidence tokens，生成后做 citation scope/freshness 校验。
- [ ] Step 21：实现 Review UI：提纲调整、逐节生成/重试、证据检查、整稿预览；不允许模型静默添加项目外论文。

**Gate：** 选 3 篇真实论文可生成对比矩阵、综述提纲和至少一节草稿；所有引用均映射到项目来源和 evidence，证据不足明确显示。

### Phase 6 — 引用与多格式导出

- [ ] Step 22：规范化 source metadata 为 CSL-JSON，建立 style registry、license attribution 和 GB/T/APA/IEEE/Chicago/MLA golden tests。
- [ ] Step 23：实现 Markdown、DOCX、CSV、JSON deterministic renderers；导出使用当前 artifact revision，文件名安全且原子写入。
- [ ] Step 24：实现 Web export preview/download；DOCX 包含标题层级、比较表、格式化引用、参考文献和 evidence appendix。

**Gate：** 同一 frozen artifact 重复导出字节/语义稳定（DOCX 忽略容器时间字段）；CSV 可重新读取；JSON 通过 schema；五种引用样式经人工样例核对。

### Phase 7 — Agent、Web 正确性与产品收尾

- [ ] Step 25：将 Agent session/transcript/pending confirmation 持久化并绑定 project；增加 project source/artifact/evidence read tools，写入和联网保持确认。
- [ ] Step 26：修复 SSE disconnect session race，接入 cancel event；实现 deadline-aware tool handlers，补齐 timeout/idempotency 合同。
- [ ] Step 27：完成 Dashboard、Evidence & Notes、job center、空状态、错误提示、中文帮助和响应式样式；HTMX/JS 全部随 wheel 本地打包。
- [ ] Step 28：增加 deterministic product scenarios、live DeepSeek/manual provider smoke、质量 rubric、隐私/安全审查和数据备份恢复演练。
- [ ] Step 29：更新 README/架构/工作流/数据模型/安全/评估文档，构建 wheel，并完成从空目录安装到 `pra serve` 的发布验收。

**Gate：** 三条核心用户旅程全部通过；服务重启不会丢项目/artifact/job/session；所有 live 结果与限制分开记录，不能用 scripted tests 宣称模型质量。

## Verification

### Automated/offline

```bash
python -m pip install -c requirements-dev.lock -e ".[dev]"
python -m pip check
python -m pytest -q
python scripts/check_tmp_space.py
python -m pip wheel . --no-deps --wheel-dir dist
python scripts/check_wheel.py dist
```

必须验证：

- 原 225 个合同测试在 package/CLI rename 后继续通过；
- schema migration、旧 Pagent dry-run/import、目标冲突和中途失败回滚；
- provider fixture parsing、DOI/arXiv/URL/content-hash dedupe、rate-limit/backoff；
- SSRF 各类地址、DNS/redirect rebinding、大小/MIME/timeout、snapshot hash/gzip；
- project/source/artifact revision/freshness、job CAS/restart/cancel；
- deep-read/compare/review 使用 scripted LLM，验证 schema、预算、引用范围、quote exactness、一次 repair 和不足证据；
- Markdown/DOCX/CSV/JSON round-trip/golden tests与五种 CSL style 样例；
- HTMX full-page/fragment、CSRF、API key、路径脱敏、SSE disconnect 和持久 session；
- wheel 内含 templates/static/CSL styles，干净 venv 安装后 `pra` 可运行。

所有 provider/LLM 自动测试默认无网络、无真实 key；live test 使用显式 marker/profile，不能混入普通 pytest。

### Manual product journeys

1. **单篇精读**：创建项目→导入真实 PDF→提出研究问题→生成九栏精读卡→展开 evidence/页码/原文→编辑一个栏目→导出 Markdown/DOCX→重启后恢复。
2. **多篇比较/综述**：选择 3 篇真实论文→补齐精读卡→生成比较矩阵→调整提纲→生成一节带证据综述→切换 GB/T/APA/IEEE→导出 DOCX/CSV/JSON。
3. **发现与入库**：同一 query 并行搜索 arXiv/Semantic Scholar/Crossref→观察 dedupe/provenance→下载一篇 PDF→抓取一个公开网页快照→两者均可 hybrid search→加入项目。
4. **新鲜度与恢复**：修改/重新索引一个来源→旧 artifact 显示 stale 且历史可读→重新生成新 revision；任务执行中重启→job 显示 interrupted/可安全重排；Agent 待确认操作刷新后仍可确认/取消。

### Real DeepSeek/provider evidence

- 使用真实 DeepSeek 跑至少 2 张单篇卡、1 个三篇比较、1 个综述 section；保存脱敏的模型、usage、finish reason、耗时、schema/prompt version、失败/repair 情况。
- 每个精读栏目人工检查“结论是否被 evidence 支持、quote 是否为原文、局限性是否被模型夸大”；综述逐 claim 检查 source/evidence 映射。
- Semantic Scholar/Crossref/arXiv live smoke 单独记录日期、query、HTTP 状态、限流/缺字段，不把 fixture 当实时可用性。
- 真实结果不可预先写死；provider 未返回 usage/finish reason 时保存 `null/unknown`。

### Security/privacy/data checks

- Git/wheel/log/API 中不得出现 DeepSeek/Semantic Scholar key、完整论文正文、原始 HTML、主机绝对路径；
- Web snapshot 只能作为文本/下载查看，禁止把原始 HTML 注入应用 DOM；
- 远程监听必须继续要求 API key + TLS，所有写表单验证 CSRF；
- UI 明示：DeepSeek 会收到用户问题与选中的文本片段；local index 不等于全文不出本机；
- citation scope/freshness 只证明来源身份与当前性，不自动证明自然语言蕴含；
- 备份 `~/.pragent` 后可在空目录恢复项目、snapshot、artifact、evidence 和索引；旧 `~/.pagent` 始终不变。

## Explicit non-goals

首个 PRAgent 产品版本不做：

- 修改 `paper-agent/` 或 `pagent-java/`；
- OCR、扫描 PDF、公式视觉理解、图片/图表/复杂表格分析；
- Ollama/本地聊天模型、多模型路由；
- pgvector/Milvus、分布式任务队列、云部署、多用户/团队协作或移动 App；
- Semantic Scholar bulk dataset、全网爬虫、需要登录/付费墙/执行 JavaScript 的网页抓取；
- Zotero 同步、BibTeX 导出或 Word 动态 citation fields（DOCX 使用静态格式化引用）；
- 自动代写整篇最终论文、查重、事实/语义蕴含的完全自动证明；
- 自动发布、覆盖用户论文文件或未经确认的 Agent 写入/联网操作。
