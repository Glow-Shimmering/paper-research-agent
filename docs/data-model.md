# PRAgent 数据模型

PRAgent 以 SQLite 为 source of truth。`papers/chunks/evidence` 继续承担全文索引与稳定证据；研究工作区通过独立 repository 关联这些兼容表，而不复制检索正文。

## Repository 边界

- `Store`：兼容的论文、分块、检索快照、evidence、Agent run/event。
- `ResearchRepository`：project、question、canonical source/provenance、project membership、artifact revision/evidence link、note。
- `JobRepository`：持久 research job、claim lease、progress、cancel 和状态 CAS。

每个 repository 使用独立 SQLite 连接。复合写入使用显式事务；跨进程竞争通过 `BEGIN IMMEDIATE` 与行 `version` compare-and-swap 协调。研究表的修改不会提高 `index_revision`，只有影响全文检索快照的 `papers/chunks/embed_model` 修改才会使搜索缓存失效。

## 核心关系

```mermaid
erDiagram
    research_projects ||--o{ research_questions : contains
    research_projects ||--o{ project_sources : selects
    research_sources ||--o{ project_sources : selected_by
    research_sources ||--o{ source_identities : identified_by
    research_sources ||--o{ source_records : observed_from
    papers o|--o| research_sources : indexed_as

    research_projects ||--o{ research_artifacts : owns
    research_sources o|--o{ research_artifacts : deep_read_target
    research_artifacts ||--o{ artifact_revisions : versions
    artifact_revisions ||--o{ artifact_evidence : cites
    evidence ||--o{ artifact_evidence : supports
    research_projects ||--o{ research_notes : owns

    research_projects o|--o{ research_jobs : scopes
    research_artifacts o|--o{ research_jobs : produces
```

## Canonical source 与 provenance

`research_sources` 是 UI/API 使用的统一来源实体。一个来源可以具有多个 `source_identities` 和多个 provider `source_records`：

- identity 类型：normalized DOI、versionless arXiv ID、canonical URL、content SHA-256；
- `(identity_kind, normalized_value)` 全局唯一；
- provider 原始 metadata 作为 JSON 保存，不因 canonical 字段选择而丢失；
- `indexed_paper_id` 可空且唯一，把来源关联到现有全文索引；
- Web snapshot 的路径只在内部表中保存，公开响应不得暴露主机路径。

`ensure_source_for_paper()` 使用论文内容 SHA-256 幂等地把现有本地论文提升为 research source，供项目工作区选择。发现层统一返回 `NormalizedSource`，只使用 DOI → versionless arXiv ID → canonical URL → content SHA-256 作为确定性 identity；标题或作者相似度绝不自动合并。任意共享 identity 都会做传递式分组，canonical metadata 由固定 provider 优先级选择，而每条 provider provenance 均保留。

聚合结果通过 `upsert_merged_source()` 在一个 `BEGIN IMMEDIATE` 事务内写入 canonical row、identities 和 provider records。后到的桥接记录若同时命中两个旧 source，repository 会确定 winner，合并 project membership、artifact/note 引用、全文关联和 provenance，再删除重复 row；任一步失败均整体回滚。Semantic Scholar API key 只存在于请求头，不进入 URL、cache envelope 或数据库 provenance。

普通网页的 raw HTML 不进入 SQLite：`snapshot_path` 只保存 content-addressed gzip 文件名，`snapshot_sha256` 校验解压内容，`extracted_text` 保存 Trafilatura 产生的纯文本。research source 的内部 `locator` 只含文档类型、最终 URL 和 snapshot hash；绝对 snapshot root 由配置持有，不通过公开序列化返回。

全文层继续使用兼容表名 `papers`，但 v3 增加的 `source_kind/canonical_uri/locator` 现在参与真实读写：PDF 的 `path` 仍是内部绝对路径，Web document 使用不可执行的 `pragent-web://<source-id>` 逻辑 locator。两类正文都进入相同 `chunks`、embedding、hybrid search 与 evidence 流程。`attach_indexed_paper()` 在一个事务内加入 content SHA identity、合并重复 canonical source 并更新 `indexed_paper_id`；跨 repository/store 无法成为单个 SQLite 连接事务，因此索引先提交、source CAS 后关联，CAS 失败时可由同一幂等入口重跑恢复。

## Artifact 与 revision

`research_artifacts` 是稳定逻辑对象；`artifact_revisions` 是不可变版本：

- artifact 的 `current_revision_number` 指向当前版本号，避免 artifact/revision 循环外键；
- 新 revision 与 artifact 当前版本在一个事务内提交；
- 人工修改、新模型结果或 repair 均追加 revision，不覆盖历史；
- `artifact_evidence` 关联到具体 revision 与字段路径，而不是只关联 artifact；
- evidence 使用 `ON DELETE RESTRICT`，保证历史 artifact 的证据链不会被静默删除。

每个 revision 保存 `source_fingerprint`。单篇 artifact 的 fingerprint 来自关联 source 的当前 PDF/content/snapshot hash；项目级 artifact 来自排序后的全部 project source。当前 fingerprint 与保存值不一致时 artifact 为 stale，旧 revision 仍可读取。

Deep Read 的模型 revision 使用更严格的原子保存入口：在同一个 `BEGIN IMMEDIATE` 中重新计算并核对生成开始时的 fingerprint，验证 evidence 属于 artifact 当前 indexed source、当前 chunk/hash 未过期、field path 属于固定九栏且 quote 是 evidence snapshot 的精确子串，然后一起提交 revision、全部 evidence links、artifact current pointer 以及 model/usage/finish reason/prompt/schema metadata。任一 forged ID、quote、跨来源 evidence 或生成期间 source 变化都会整体回滚。

Project-level `comparison` artifact 同样使用专用原子保存入口。其 content 固定所选 2–20 个 project source、比较维度和完整 cell 矩阵；保存时验证项目 fingerprint、全部 source membership/index 状态，以及每条 cell evidence 是否属于声明来源且仍与当前 chunk/hash/quote 一致。content 中的 evidence refs 必须与待写入 `artifact_evidence` links 完全一致，防止通过省略 link 保存表面有引用但不可审计的矩阵。

比较 cell 的人工编辑不会原位更新 content，而是用 artifact version 做 CAS 后追加 `created_by=user` revision。保留证据时重新验证全部 evidence；显式设为 `insufficient_evidence` 时，该 cell 的 refs 与 links 必须同时为空。项目 fingerprint 变化后 artifact 为 stale，历史仍可读取但不能继续派生人工 revision。

## Notes

`research_notes.scope_kind` 明确限定三种范围：

- `project`：不带 source/evidence；
- `source`：只带 source；
- `evidence`：只带 evidence。

Repository 同时验证 source 已加入当前 project；数据库 `CHECK` 和外键作为第二层防线。

## Jobs

`research_jobs` 状态：

```text
queued → running → succeeded | failed | cancel_requested | cancelled | interrupted
interrupted → queued | cancelled | failed
```

Job 保存 payload/result、进度、attempt/max-attempt、priority/run-after、timeout、lease、取消时间和错误。`idempotency_key` 使用部分唯一索引；同一个 key 只有在任务语义参数一致时才返回已有记录，否则 fail closed。Claim 与状态更新同时检查 status 和 `version`，避免两个 worker 重复占有同一任务。

后台 worker/queue 复用这些持久化合同，不在 HTTP 请求内执行长任务。Web 启动恢复在一个事务内把遗留 `running` 标记为 `interrupted`，把遗留 `cancel_requested` 终结为 `cancelled`，并且只重排 `idempotent=1 AND attempts < max_attempts` 的任务。固定 worker 使用 lease owner + row version 更新进度和终态；过期 lease 同样只允许幂等且仍有额度的任务重排。排队/中断任务的取消直接终结，运行中任务则在 handler 的下一阶段边界协作取消。
