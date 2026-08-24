# PRAgent 产品工作流

本文记录当前已实现的 Phase 2–3 Web 工作流；深读卡、比较、综述与导出仍属于后续 Phase，不在这里提前宣称可用。

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

## 5. JSON API

新能力位于 `/api/v1`：

- `POST /api/v1/discover/search`：多 provider 聚合、持久化与部分失败；
- `GET /api/v1/sources`：安全分页来源目录；
- `POST /api/v1/sources/web`：显式抓取并索引网页；
- `POST /api/v1/sources/{source_id}/download`：下载具有 arXiv ID 的 PDF 并索引；
- `POST /api/v1/projects/{project_id}/sources/{source_id}`：选择 canonical source 进入项目。

默认仅允许 loopback；远程模式继续要求 API key + TLS。所有 HTMX 写表单要求同源和 double-submit CSRF token，JSON 写请求受 API key/origin/body limit 中间件保护。
