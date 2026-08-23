# Paper Research Agent（PRAgent）

## 项目说明

证据优先的本地论文研究 Agent：整理 PDF 资料库，以受控工具调用完成检索、深读、问答与笔记。

- 索引本地 PDF 论文库（增量：重复运行只处理变化的文件）
- 混合检索：BM25 关键词 + 本地语义向量（fastembed，CPU 推理），RRF 融合
- 问答：OpenAI 兼容 API（DeepSeek / OpenAI / 通义等），未配置 key 时退回纯检索；CLI、TUI 与 Web 均支持流式输出；回答“库里有哪些论文”类问题时以注入的论文库目录为权威来源（正文中的参考文献不算库藏）
- 受控 Agent：显式 run 状态、调用预算、工具效果分类、写入/联网确认与可恢复执行
- Web 端 Agent：SSE 流式对话、实时工具调用卡片、可视化确认票据、证据高亮与 run 审计侧栏
- 持久研究项目：创建 project、编辑/排序研究问题、从现有本地论文库选择来源，刷新或重启后恢复
- 证据链：单篇检索、页面/相邻分块深读、稳定 evidence ID、固定证据与引用校验
- 可审计：Agent run 与结构化事件持久化；内置 37 个无网络、确定性的状态机/引用合同场景
- CLI 与本地 Web 界面双入口（统一命令名 `pra`）
- 索引过程只读：解析元数据建库浏览，不改动已有 PDF；下载工具会新增或原子替换同 arXiv 编号文件

设计细节见 [架构说明](docs/architecture.md)。

## 安装

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
```

也可构建并安装普通 wheel；Web 静态资源会一并打包：

```powershell
.venv\Scripts\python -m pip wheel . --no-deps --wheel-dir dist
$wheel = Get-ChildItem .\dist\paper_research_agent-*.whl | Sort-Object LastWriteTime -Descending | Select-Object -First 1
.venv\Scripts\python -m pip install $wheel.FullName
pra --version
pra serve
```

## 配置

复制 `.env.example` 为 `.env` 并填写（环境变量统一使用 `PRA_` 前缀）：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `PRA_ENV_FILE` | （未设） | 显式指定配置文件；相对路径按当前工作目录解析 |
| `PRA_LLM_BASE_URL` | `https://api.deepseek.com` | OpenAI 兼容 API 地址 |
| `PRA_LLM_API_KEY` | （空） | 留空则问答退回纯检索 |
| `PRA_LLM_MODEL` | `deepseek-chat` | 问答模型名 |
| `PRA_WEB_API_KEY` | （空） | Web API key；监听非本机地址时必须设置 |
| `PRA_EMBED_MODEL` | `BAAI/bge-small-zh-v1.5` | 嵌入模型（首次索引联网下载 ~100MB） |
| `PRA_DATA_DIR` | `~/.pragent` | 独立数据目录；不会自动读取或修改旧 Pagent 数据 |
| `PRA_DOWNLOAD_DIR` | （未设） | 对话下载目录；不设时使用显式配置的 `PRA_DATA_DIR`，否则用已索引论文库目录 |
| `PRA_NOTE_DIR` | `PRA_DATA_DIR/notes` | 笔记保存目录（`save_note` 工具与 `/export` 命令） |

配置文件查找顺序为：`PRA_ENV_FILE`、当前工作目录的 `.env`、editable 源码仓库根目录的 `.env`。显式指定但文件不存在时不会回退到其他 `.env`；非空系统环境变量始终优先。普通 wheel 安装后建议在运行目录放置 `.env`，或设置 `PRA_ENV_FILE`。

> 首次 `pra index` 需联网下载嵌入模型（~100MB，缓存于本地）。国内网络直连 HuggingFace 常失败：在 `.env` 中设置 `HF_ENDPOINT=https://hf-mirror.com` 与 `HF_HUB_DISABLE_XET=1`（镜像不支持 Xet 存储，缺一不可）。

## 用法

```powershell
pra index            # 索引当前目录（可加参数指定目录，如 pra index .\papers）
pra list                     # 浏览论文库
pra search "注意力机制"      # 混合检索
pra websearch "llm survey"   # 联网检索 arXiv 论文（英文效果更佳）
pra ask "这篇论文提出了什么方法？"   # 问答（需 API key，默认流式输出；--no-stream 关闭）
pra ask --web "问题"         # 问答时同时联网检索 arXiv 论文
pra chat                     # 受控 Agent TUI：需要 API key；回答逐字流式渲染，联网/写操作需 /confirm
pra serve                    # 启动 Web 界面 http://127.0.0.1:8000（研究项目 + 兼容工作台）
pra status                   # 库与配置状态
pra import-pagent --source ~/.pagent          # 只读检查旧 Pagent 数据（默认 dry-run）
pra import-pagent --source ~/.pagent --execute # 校验后复制导入到 PRA_DATA_DIR
pra --version                # 显示当前版本（版本号与 wheel 元数据同源）
```

`pra import-pagent` 绝不原地升级 `~/.pagent`：默认只读检查 schema、SQLite 完整性、论文内容哈希和复制清单。`--execute` 使用 SQLite online backup（包含已提交 WAL）复制到目标同盘 staging，在 staging 中迁移并校验后再原子落位；目标目录已存在、旧文件变化或任一步失败都会拒绝覆盖。位于旧数据目录内的论文路径会重写到新目录，外部论文路径经校验后保留原引用。

Windows 下 CLI 会把标准输出和错误输出配置为 UTF-8，含数学符号等 Unicode 文本的检索结果可以直接输出或重定向到 UTF-8 文件。

`pra serve` 默认只监听本机。主页保留原检索/问答/Agent/论文库兼容工作台，并提供「研究项目」入口；project/question/source membership 全部写入 SQLite，页面刷新与服务重启后仍可恢复。研究工作区使用服务端 Jinja 模板与 wheel 内置 HTMX，写表单采用 SameSite double-submit CSRF token；返回的本地论文项不暴露主机绝对路径。

使用 `--host 0.0.0.0`、非 loopback Host 或 HTTPS 反向代理时必须设置 `PRA_WEB_API_KEY`；Web 页面会在首次 API 请求时询问 key，并仅在当前浏览器会话中保存。直接远程监听还必须同时传入 `--ssl-certfile` 与 `--ssl-keyfile`，或让 HTTPS 反向代理转发到 `127.0.0.1`（并正确转发原始 scheme/host）。仅在可信隔离网络中，才可显式添加 `--allow-insecure-http` 使用明文 HTTP。

### 隐私边界

PDF 解析、分块、BM25 和向量嵌入均在本地执行。`pra ask`、`pra chat` 和 `pra index --refine` 会把问题以及命中的论文片段或首页文本发送给 `PRA_LLM_BASE_URL` 指向的第三方服务；`--web`/`websearch` 会把查询词发送给 arXiv。敏感论文应使用可信的自托管兼容接口，或使用 `--no-llm` 纯检索模式。

联网检索基于 arXiv API（免费、无需 key，遵守 3 秒请求间隔）；Web 界面在检索/问答页勾选「联网（arXiv）」即可。Web 的「Agent」标签页与 `pra chat` 共享同一受控运行时：工具调用实时展示，联网或写入操作会弹出待确认卡片（含参数摘要与绑定摘要），点击「确认执行」后沿原始 `tool_call_id` 续跑，或点击「取消」终止；每次对话都会生成可审计的 run，可在侧栏回放结构化事件时间线。

`pra chat` 为终端 Agent 界面（textual）。每个用户问题会创建一个持久 run，并按 `proposed → running → awaiting_confirmation → succeeded/failed/cancelled/blocked` 状态推进；轮次、工具调用、外部调用和引用修复都有硬预算。结构化事件只保存必要元数据、哈希、状态和结果摘要，便于定位失败与回放。

模型可使用本地混合检索、单篇检索、分页概览、PDF 页面阅读、相邻分块阅读、固定/读取证据、论文列表、库状态、arXiv 搜索、下载、索引和笔记工具。读取本地资料可自动执行；联网或本地写入不会立即执行，必须先检查绑定了参数摘要与 SHA-256 的确认票据，再输入 `/confirm`。确认完成后 Agent 会沿原始 `tool_call_id` 自动续跑；`/cancel` 会记录取消结果并结束对应 run。Agent 不能切换论文库根目录；切换目录必须在终端显式运行 `pra index <目录> --force`。

深读工具返回稳定的 `[E:ev_…]` 证据标记。Agent 的最终回答只能引用本轮真实返回的 evidence ID；索引中文件或分块发生变化后，已固定证据会标为 stale，而不是静默指向新内容。传统 `pra ask` 继续使用 `[n]` 引用，两条问答路径共享同一引用验证器。SQLite 使用有序 migration 升级到 schema v5；已有磁盘库升级前会生成一致备份，任一步失败则回滚整个 migration 事务并保留原论文与分块。

对话内支持 `/help`、`/clear`、`/copy`、`/export`、`/confirm`、`/cancel`、`/quit`。

## 构建与测试

Python 3.11 可使用已验证的完整版本约束集复现开发环境；3.10/3.12 由 CI 验证当前可解析依赖组合的兼容性：

```powershell
.venv\Scripts\python -m pip install -c requirements-dev.lock -e ".[dev]"
.venv\Scripts\python -m pip check
.venv\Scripts\python -m pytest -q
.venv\Scripts\python scripts\check_tmp_space.py
.venv\Scripts\python -m pip wheel . --no-deps --wheel-dir dist
.venv\Scripts\python scripts\check_wheel.py dist
```

若需要连同真实本地嵌入模型做一次端到端冒烟（首次可能联网下载模型），运行：

```powershell
.venv\Scripts\python scripts\smoke_real.py
```

GitHub Actions 会在 Windows/Python 3.11 与 Ubuntu/Python 3.10、3.11、3.12 上执行测试，并在 3.11 上检查约束依赖与 wheel 资源。

37 个 Agent 场景使用版本化 JSON 和脚本化 LLM/工具结果，不访问网络；它们用于回归状态机、预算和结构化引用合同，覆盖成功、拒答、引用错误、确认/取消、工具失败、重试和预算熔断等路径。真实 `chat_turn` 与工具确认边界由 `tests/test_chat.py`、`tests/test_tools.py` 的集成测试覆盖。该确定性场景集不调用真实模型，也不用于宣称模型具备提示注入抵抗能力，或引用证据与回答论断之间已经通过语义蕴含验证。
