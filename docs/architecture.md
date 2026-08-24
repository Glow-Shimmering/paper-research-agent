# Paper Research Agent（PRAgent）架构说明

本文描述派生基线 `v0.1.0` 的运行边界与关键设计。PRAgent 是一个证据优先的本地论文研究 Agent：PDF、索引和证据默认保存在本地，模型负责决策与生成，应用代码负责权限、状态、预算和引用约束。

## 组件

```mermaid
flowchart LR
    User["用户"] --> TUI["Textual TUI"]
    User --> Web["FastAPI / Web"]
    TUI --> Runtime["Agent Runtime"]
    Web --> Ask["检索问答"]
    Web --> Workspace["Project Workspace / HTMX"]
    Workspace --> ResearchRepo["ResearchRepository"]
    ResearchRepo --> Store
    Runtime --> Policy["ToolSpec + Policy"]
    Runtime --> LLM["OpenAI-compatible LLM"]
    Policy --> Tools["检索 / 深读 / 下载 / 索引 / 笔记"]
    Tools --> Store["SQLite Store"]
    Tools --> PDF["本地 PDF"]
    Tools --> Arxiv["arXiv"]
    Store --> Evidence["Evidence + Run + Event"]
```

- `agent.py`：显式状态机、Planner、Executor、预算和统一引用验证。
- `chat.py`：模型与工具的编排循环、暂停/恢复、结构化事件和最终答案验证。
- `tool_protocol.py` / `tools.py`：工具合同、参数校验、副作用分类、确认票据和执行结果。
- `store.py`：论文、分块、向量、稳定证据、Agent run 和事件日志的 SQLite 兼容门面。
- `storage/migrations.py`：从 v1 开始的有序 schema migrations、迁移历史校验、磁盘库一致备份与事务回滚边界。
- `storage/research_repository.py`：project/source/provenance/artifact revision/evidence link/note 的事务、分页与版本 CAS。
- `storage/job_repository.py`：持久 job、幂等 enqueue、lease claim、progress/cancel/status CAS。
- `web/routes/projects.py`：`/api/v1` project JSON API 与 Jinja/HTMX project/question/source vertical slice。
- `sources/base.py` / `sources/identity.py`：provider-neutral normalized source contract，以及 DOI → arXiv ID → canonical URL → content SHA 的确定性、可传递合并。
- `sources/arxiv.py`：有界 arXiv Atom adapter；`websearch.py` 仅保留旧调用合同的兼容门面。
- `sources/semantic_scholar.py` / `sources/crossref.py`：字段白名单 adapter、可选认证与 provider 原始记录规范化。
- `sources/http.py` / `sources/discovery.py`：不缓存请求密钥的 fixture/response cache、线程安全节流、429/5xx 有界退避，以及多 provider 聚合与部分失败隔离。
- `ingestion/safe_fetch.py`：逐跳 URL/DNS 校验并把连接固定到已验证公网 IP；限制重定向、总超时、MIME 与响应大小。
- `ingestion/snapshots.py` / `html_extract.py` / `web.py`：原始 HTML 的 content-addressed gzip 原子快照、Trafilatura 文本抽取及可恢复 source 状态。
- `ingestion/indexing.py`：把网页纯文本或已下载 PDF 适配为通用 `Paper(source_kind/canonical_uri/locator)`，再复用同一 chunk/embed/index/source-link 事务边界。
- `search.py`：PDF 与 Web document 共用 BM25、本地向量、RRF、一致 snapshot cache 和稳定 evidence。
- `tui.py`：终端 Agent 入口（回答逐字流式渲染）；Web 保留检索、SSE 问答与受控 Agent 兼容工作台，并增加持久研究项目入口。

## Run 生命周期

```mermaid
stateDiagram-v2
    [*] --> proposed
    proposed --> running
    proposed --> cancelled
    proposed --> blocked
    running --> awaiting_confirmation
    awaiting_confirmation --> running
    awaiting_confirmation --> cancelled
    awaiting_confirmation --> failed
    awaiting_confirmation --> blocked
    running --> succeeded
    running --> failed
    running --> cancelled
    running --> blocked
```

每次 TUI 问题创建独立 run。轮次、工具调用、外部调用、重规划和引用修复均受硬预算限制；终态不可恢复，非法状态迁移会被拒绝。

## 工具与审批边界

每个工具必须通过 `ToolSpec` 显式声明：

- JSON Schema 参数合同；
- `READ_LOCAL`、`WRITE_LOCAL`、`NETWORK` 副作用；
- 超时和幂等元数据；
- 结构化 `ToolResult`，包括错误码、重试属性和 evidence ID。

写入或联网操作不会立即执行。运行时保存包含工具名、冻结参数、`tool_call_id` 和参数摘要的确认票据；用户批准后只允许执行该票据绑定的动作。取消会补齐原工具协议消息并把 run 转为 `cancelled`。状态 CAS 失败时不清理票据，允许安全重试。

## 证据一致性

检索和深读工具返回稳定的 `[E:ev_…]` 标识。证据保存来源论文哈希、分块文本哈希、页码和快照正文：

- 最终回答只能引用本轮工具实际返回且未过期的 evidence ID；
- PDF 或分块变化后，旧证据保留用于审计，但被标记为 stale，不能进入引用白名单；
- 页面深读前验证磁盘 PDF 哈希与索引一致，避免新正文错误绑定旧证据；
- `paper ask` 的 `[n]` 与 Agent 的 `[E:…]` 使用同一验证模块。

这里验证的是引用身份、范围和新鲜度，不等同于自动证明每个自然语言论断与证据之间存在语义蕴含。

## 持久化与可观测性

SQLite schema v5 保留 v1/v2 的论文索引、证据与 Agent 审计表，并为研究工作区建立持久化边界：

- v1：`papers/chunks/meta`；
- v2：`evidence/agent_runs/agent_events`；
- v3：通用 document locator、project、research question、canonical source/identity、provider record 与 project-source membership；
- v4：artifact、不可变 revision、artifact-evidence link 与 research note；
- v5：可恢复 job、Agent session/transcript 与冻结的 pending action。

每个 migration 都有名称、版本和 SHA-256 checksum 记录。磁盘数据库只在声明版本和已有表结构通过检查后迁移；升级前使用 SQLite backup API 生成一致备份，所有待执行步骤位于同一个 `BEGIN IMMEDIATE` 事务中。任一步、外键检查或历史校验失败都会回滚，未来版本数据库则不做修改并拒绝打开。研究对象与 job 使用独立行 `version` 做 compare-and-swap，不复用只服务于搜索缓存失效的 `index_revision`。完整关系与 freshness 规则见 [数据模型](data-model.md)。

事件默认保存必要元数据、哈希和结果摘要，避免把完整论文正文作为 trace 复制。LLM 响应同时保留 usage、finish reason 和 response ID，供后续成本与延迟评测使用。

### Web 项目工作区边界

`/ui/projects` 使用服务端 Jinja autoescape 与 wheel 内置 HTMX 2.0.8（MIT license 随资源打包），不依赖 Node/CDN。写表单必须同时通过同源检查、1MB body limit 和 HttpOnly/SameSite double-submit CSRF cookie；远程模式先通过 `X-PRA-Key` 换取不含原始 key 的 HttpOnly UI cookie。项目来源响应只返回题录、状态和安全 filename，不返回 `papers.path`、snapshot path 或抽取正文。project、question 与 source membership 均来自 SQLite repository，页面刷新或服务重启不依赖进程内状态。

完整网页威胁模型、DNS pinning、snapshot 与 raw HTML 边界见 [来源抓取安全](source-security.md)。

### 旧 Pagent 显式导入

`import_pagent.py` 只接受已验证的 Pagent schema v1/v2，默认 dry-run，且从不在旧目录上运行 migration。执行导入时先建立文件 hash 清单，再通过 SQLite online backup 把包含已提交 WAL 的一致快照写入目标同盘 staging；路径重写、v5 migration、计数/外键/quick-check 和文件二次 hash 均在 staging 完成。只有全部通过后才原子重命名为目标目录，目标已存在或中途失败均 fail closed。

## 测试边界

- 单元与集成测试覆盖工具合同、索引与证据一致性、确认/取消、CAS 冲突和 TUI 续跑。
- 37 个离线 JSON 场景用于回归状态机、预算和引用合同。
- 场景使用脚本化模型与工具结果，不代表真实模型质量、提示注入抵抗能力或语义蕴含评测。

下一阶段会在此基础上增加真实任务 benchmark、成本面板、多来源检索与笔记双链。
