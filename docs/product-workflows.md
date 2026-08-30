# PRAgent 产品工作流

本文记录当前已实现的研究工作流：Phase 2–5 的项目/发现/精读/比较/综述、Phase 6 的引用规范化与多格式导出，以及研究工作台的产品化页面（Dashboard、任务中心、证据与笔记、帮助）。所有描述均以已合入代码与测试为准。

## 0. Dashboard、任务中心与帮助

- `/ui/`（Dashboard）：项目、最近来源、进行中任务与库统计的统一入口；每个区块都有中文空状态引导（例如来源库为空时提示去 Discover 或 `pra index`）。
- `/ui/jobs`（任务中心）：全局任务列表，按状态筛选，HTMX 每 3 秒轮询刷新进度与失败信息；queued/running/interrupted 任务可请求取消（版本 CAS，冲突显式 409）。取消只保证在阶段边界停止，不会中断进行中的模型请求。
- `/ui/help`：三条核心工作流、任务与恢复说明，以及隐私与安全边界（云端模型仅接收问题与选中文本；引用/新鲜度校验不等于语义蕴含）。

## 1. 创建研究项目

1. 从兼容工作台进入「研究项目」，或直接打开 `/ui/projects`；
2. 创建项目并填写主题/描述；
3. 在 workspace 新建、编辑、排序或删除研究问题；
4. 从已索引本地文档中选择来源，或先到 Discover/Library 获取来源。

Project、question 与 project-source membership 均保存在 SQLite。HTMX fragment 和完整页面使用同一 repository，不依赖浏览器或进程内状态。

## 2. 聚合发现并去重

1. 打开 `/ui/discover`，填写 query；
2. 选择 arXiv、Semantic Scholar、Crossref 中的一个或多个 provider；
3. 提交后每个 provider 独立执行有界请求。单个 provider 失败会显示错误和 retryable 状态，但不丢弃其他结果；
4. 结果只按 DOI → versionless arXiv ID → canonical URL → content SHA 合并，绝不因为标题/作者相似而自动合并；
5. provider badges 与“合并 N 条记录”标识展示 provenance；canonical row 和所有 provider raw metadata 在一个 repository 事务中持久化；
6. 题录可直接加入选定项目。具有 arXiv ID 的来源可由用户点击「下载并索引 PDF」。

Semantic Scholar key 是可选请求头；不会进入 URL、response cache、SQLite provenance 或 UI。Crossref 可设置联系邮箱进入 polite pool。

## 3. 导入普通网页

1. 在 Discover 的「导入普通网页」中显式提交公网 HTTP(S) URL；
2. fetcher 对初始 URL和每次 redirect 重新解析所有 IP，拒绝任何非公网地址，并连接到已校验的 pinned IP；
3. 错误 MIME、过大正文、超时、私网 redirect 或抽取失败会把 source 标为 `failed`，Library 显示稳定错误码；
4. 成功响应以 content SHA 命名保存 gzip snapshot，Trafilatura 只把纯文本与规范化 metadata 写入 SQLite；
5. 抽取文本立即走现有 chunk/embed/index 流程，可选择同时加入项目。

失败 source 可在 Library 重试。重试仍执行完整安全检查；成功后清除旧 `last_error`。原始 HTML 不在 UI 中渲染，公开 API 不返回 snapshot 文件名或抽取全文。详细威胁模型见 [来源抓取安全](source-security.md)。

## 4. 统一来源库与全文检索

`/ui/library` 可按关键词、paper/web 类型和 discovered/fetching/ready/failed/archived 状态筛选：

- `discovered`：已有题录，尚未获取全文；
- `fetching`：正在执行显式抓取或下载；
- `ready`：来源可用；`indexed=true` 进一步表示全文已进入 chunks；
- `failed`：上次操作失败，可重试；
- `archived`：用户归档。

已下载 PDF 与 Web extracted text 都写入兼容 `papers/chunks` 底座，并带 `source_kind/canonical_uri/locator`。同一次 `/api/search` 或 `hybrid_search()` 可以命中两种文档，随后都可固定为稳定 evidence。公开 Web 响应对 PDF 只显示 filename，对 Web 只显示 canonical URL；主机绝对路径、snapshot locator、provider raw metadata 与全文留在 storage boundary 内。

## 5. 单篇精读

1. 在项目 workspace 进入「单篇精读」，选择一个已完成全文索引的项目来源；
2. 生成动作只负责写入持久 job，固定 worker 在后台按九个字段分别检索、map，再 reduce 为严格 `DeepReadCard`；页面通过 HTMX polling 展示 0–9 进度；
3. 九栏固定为研究问题、相关工作、核心方法、创新点、数据集与实验、主要结果、局限性、未来工作、关键原文证据；
4. 每栏可展开 evidence drawer，查看页码、精确原文 quote 和当前 chunk 上下文；公开响应不包含主机路径、paper/chunk ID、snapshot locator 或 raw metadata；
5. 「人工编辑」和「重新生成本栏」都会新增不可变 revision，不覆盖历史。单栏重生成只允许在当前卡未 stale 时执行，避免把八个旧字段错误标记为最新；
6. 来源正文变化后旧卡显示 stale，但历史 revision 与旧 evidence snapshot 保留可读；完整重新生成使用新的 source fingerprint。

模型输出在保存前验证 Pydantic schema、evidence 来源范围、当前 freshness 和 quote 精确子串；全流程最多进行一次 JSON repair。模型名、usage、finish reason、prompt/schema version 随 revision 保存。

## 6. Project-scoped 比较矩阵

比较工作流只接受当前 project 明确选择的 2–20 个不重复来源。每个来源必须具有 `ready` 且未 stale 的当前精读卡；缺失与过期来源分别返回稳定列表，供下一步 UI 先排队补齐，而不是让 Agent 从整库自行挑选论文。

默认九个比较维度直接复用精读卡字段，不再次调用模型。自定义维度按“一个维度覆盖全部所选来源”执行有界 LLM 调用；输出必须为完整 source × dimension 矩阵，并且只能使用对应来源精读卡已有的 evidence ID 和逐字 quote。全流程最多一次 JSON repair。

比较生成已注册为 SQLite-backed `comparison` worker job。Web/API 创建入口会先检查精读卡；缺失或 stale 时只排队补齐对应精读任务，全部 ready 后再次提交才创建比较 job。默认维度不调用 LLM，自定义维度仍使用 worker 中配置的可审计 LLM。

`/api/v1/projects/{project_id}/comparisons` 和 `/ui/projects/{project_id}/comparisons` 提供 project-scoped 创建与列表；详情按来源为行、维度为列展示完整矩阵，并可逐 cell 展开 evidence。每次人工编辑都要求 artifact version CAS，追加 `created_by=user` revision 而不覆盖历史；编辑可保留原证据，或显式切换为 `insufficient_evidence` 并原子移除该 cell 的 evidence links。来源变化后矩阵标记 stale，禁止继续编辑，但历史 revision 与 evidence 仍可读取。

保存每个 revision 时在一个事务内重新检查 project fingerprint、来源 membership、索引关联、evidence 新鲜度、evidence 所属来源、quote 原文子串，以及 JSON content 与 evidence links 完整一致。

## 7. 证据约束的综述提纲

`review_outline` workflow 接受当前项目的 1–20 个研究问题、与比较矩阵顺序完全一致的 2–20 个来源，以及一个 ready、未 stale 的当前 comparison artifact。LLM 只生成提纲标题、章节和 planned claims；研究问题快照、来源列表、comparison artifact/revision provenance 由系统写入，模型不能修改。

每个有证据的 planned claim 必须为其声明的每个 source 提供该 source 在绑定 comparison revision 中已有的 evidence ID 和逐字 quote；无法支持的计划必须显式 `insufficient_evidence=true` 且不带 refs。输出使用严格 Pydantic schema、上下文/token/call 预算和全流程最多一次 JSON repair。

保存使用专用原子入口：同一事务重新验证研究问题文本/version、完整 project fingerprint、comparison 当前 revision、所选来源 membership/index，以及 evidence 确实出现在绑定的 comparison、属于声明来源、chunk/hash 当前且 quote 为原文子串。提纲已注册为持久 `review_outline` worker job；创建与调整 UI 属于 Step 21。

`review_section` workflow 只接受当前 ready、未 stale 的 review outline 及其中一个 section。章节正文是有序 claim 列表；每条 claim 使用结构化 `source_id + evidence_id + exact quote` citation tokens，或者显式 `insufficient_evidence`。模型返回后再次检查 outline revision/freshness，随后保存事务把 token 限制在该 section planned claims 的证据集合，并再次验证来源、chunk/hash 与 quote。

章节人工编辑以 artifact version CAS 追加 `created_by=user` revision；默认保留原 citation tokens，也可以显式切换为证据不足并同时清空 refs/links。章节生成已注册为持久 `review_section` worker job；交互页面与确定性正文/引用导出均已完成。

Review Web/API 从 ready、未 stale 的 comparison 自动取得来源集合，用户不能在提纲或章节请求中静默加入项目外论文。页面支持多研究问题选择、提纲章节标题/目标调整、逐节生成与显式重试、claim evidence drawer、提纲 revision 历史和当前章节的整稿预览。提纲或绑定问题/比较/来源漂移后页面只保留历史读取，禁止继续派生章节。

## 8. 项目证据与研究笔记

`/ui/projects/{project_id}/evidence` 集中呈现项目内产物引用的证据与研究笔记：

- 证据列表聚合每个 artifact 当前 revision 的 evidence links（跨精读/比较/提纲/章节），展示 evidence ID、字段路径、页码与来源定位；网页来源显示 canonical URI，PDF 只显示文件名，不返回主机绝对路径；来源更新后对应条目显示 stale 徽标；
- 研究笔记支持项目级与来源级两种 scope：创建后可继续编辑（标题/Markdown 内容），每次保存都是版本 CAS（`expected_version`），冲突返回 409，不覆盖历史；JSON API 位于 `/api/v1/projects/{id}/notes`；
- 笔记内容按纯文本转义渲染，不执行 Markdown 内嵌 HTML，与其他页面共享同一 CSRF 与脱敏边界。

## 9. 引用元数据与样式

当前 `ResearchSource` 的 canonical metadata 会在导出边界确定性转换为 CSL-JSON，不修改数据库原始记录，也不凭空补齐缺失字段。转换包含题名、作者、年份、DOI、URL、访问日期以及期刊、卷期、页码、出版社和语言；英文空格姓名按 given/family 拆分，无法可靠拆分的姓名保留为 CSL `literal`。

内置 style registry 固定公开 `gb-t-7714-2015-numeric`、`apa-7`、`ieee`、`chicago-author-date` 和 `mla` 五个键。每个 CSL 文件加载时执行 schema 校验，未知样式或 processor 失败都会明确报错，不回退到另一个格式。样式来自 Citation Style Language 官方 styles 仓库，版本、原始链接和 CC BY-SA 3.0 归因随 wheel 保存在 `pragent/styles/ATTRIBUTION.md`。

当前导出服务先冻结 artifact 的当前 immutable revision、项目/来源版本、freshness、provider provenance 和 evidence snapshots，再以同一 frozen snapshot 生成 Markdown、DOCX、CSV 或 JSON。Markdown/DOCX 将结构化 citation tokens 交给一个共享 CSL processor context，保证数字引文与参考文献编号一致；不会把内部 token 原样泄漏进正文。

JSON 包含固定 export schema version、artifact/revision/model metadata、freshness、canonical source/CSL-JSON、identities、provider records 和 evidence snapshots。CSV 总是输出可回读的来源目录，comparison 另输出 long-form matrix。DOCX 使用真实标题层级、固定表格几何、参考文献和 evidence appendix，并把 OOXML ZIP 时间归一化以获得稳定字节。

落盘文件名由受限标题、artifact ID 和 revision number 组成，拒绝 renderer 提供的不安全后缀；临时文件在目标目录写完、flush/fsync 后使用 `os.replace` 原子替换。导出期间若 artifact、来源或 freshness 发生变化，冻结过程明确冲突失败，不混合两个 revision 的数据。

Web 的“预览与导出”页同步生成受 200,000 字符上限约束的 Markdown 预览；正式文件进入持久 `export` job。排队 payload 冻结 expected artifact revision、citation style、source versions/fingerprint 和 review section revisions，worker 开始时全部复核后才写入以 job ID 隔离的目录。下载接口只允许读取 succeeded job result 明确列出的 basename，不接受任意路径。

对 `review_outline` 导出会按提纲顺序聚合所有绑定该 current outline revision 的 current `review_section`；已生成章节使用人工/模型最新 revision 正文和结构化 citation tokens，未生成章节明确标注并保留提纲计划。Markdown/DOCX 因而提供带格式化引文、参考文献与跨提纲/章节 evidence appendix 的整稿，comparison DOCX 继续输出固定几何的比较表。

HTMX 写入仍使用 double-submit CSRF。重复 checkbox 字段由受限 URL-encoded parser 保留为列表，既不丢失多研究问题选择，也继续受 1 MB/100 字段上限约束。

## 10. JSON API

新能力位于 `/api/v1`：

- `POST /api/v1/discover/search`：多 provider 聚合、持久化与部分失败；
- `GET /api/v1/sources`：安全分页来源目录；
- `POST /api/v1/sources/web`：显式抓取并索引网页；
- `POST /api/v1/sources/{source_id}/download`：下载具有 arXiv ID 的 PDF 并索引；
- `POST /api/v1/projects/{project_id}/sources/{source_id}`：选择 canonical source 进入项目；
- `POST /api/v1/projects/{project_id}/sources/{source_id}/deep-reads`：排队生成单篇精读；
- `GET /api/v1/projects/{project_id}/deep-reads/{artifact_id}`：读取当前或历史九栏卡；
- `GET /api/v1/projects/{project_id}/jobs/{job_id}`：读取脱敏的后台进度；
- 字段 regeneration/edit、revision history 与 field evidence 使用同一 `/api/v1/projects/.../deep-reads/...` scope。

默认仅允许 loopback；远程模式继续要求 API key + TLS。所有 HTMX 写表单要求同源和 double-submit CSRF token，JSON 写请求受 API key/origin/body limit 中间件保护。

## 11. Project-scoped Web Agent

Agent 会话可在第一次提问时绑定一个研究项目，随后不能切换到其他 project；每个 run 同时带有 project/session scope。`list_project_sources`、`list_project_artifacts` 和 `list_project_evidence` 只读取该绑定项目，且不返回 snapshot path、抽取全文或服务器绝对路径。它们属于本地只读工具，不触发确认；`web_search`、下载、索引、固定证据和保存笔记等联网/写入工具继续按 effects 强制确认。

浏览器用 `GET /api/agent/sessions/{session_id}` 恢复用户/assistant/tool 卡片、project、run 和待确认卡片。数据库在一个事务中保存完整 OpenAI-compatible transcript 与冻结票据；服务重启后会复核参数 SHA-256 和 action digest，再允许按原 `tool_call_id/run_id` 确认或取消。确认执行前先用 CAS 认领票据，因此并发服务不能重复执行同一写入；页面提交不同 project ID 会返回冲突。
