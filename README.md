# paper-agent

## 项目说明

论文整理与检索 Agent：整理本地论文资料库，提供检索与问答能力。

- 索引本地 PDF 论文库（增量：重复运行只处理变化的文件）
- 混合检索：BM25 关键词 + 本地语义向量（fastembed，CPU 推理），RRF 融合
- 问答：OpenAI 兼容 API（DeepSeek / OpenAI / 通义等），未配置 key 时退回纯检索
- CLI 与本地 Web 界面双入口
- 索引过程只读：解析元数据建库浏览，不改动已有 PDF；下载工具会新增或原子替换同 arXiv 编号文件

## 安装

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
```

也可构建并安装普通 wheel；Web 静态资源会一并打包：

```powershell
.venv\Scripts\python -m pip wheel . --no-deps --wheel-dir dist
$wheel = Get-ChildItem .\dist\paper_agent-*.whl | Sort-Object LastWriteTime -Descending | Select-Object -First 1
.venv\Scripts\python -m pip install $wheel.FullName
paper --version
paper serve
```

## 配置

复制 `.env.example` 为 `.env` 并填写：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `PAPER_ENV_FILE` | （未设） | 显式指定配置文件；相对路径按当前工作目录解析 |
| `PAPER_LLM_BASE_URL` | `https://api.deepseek.com` | OpenAI 兼容 API 地址 |
| `PAPER_LLM_API_KEY` | （空） | 留空则问答退回纯检索 |
| `PAPER_LLM_MODEL` | `deepseek-chat` | 问答模型名 |
| `PAPER_WEB_API_KEY` | （空） | Web API key；监听非本机地址时必须设置 |
| `PAPER_EMBED_MODEL` | `BAAI/bge-small-zh-v1.5` | 嵌入模型（首次索引联网下载 ~100MB） |
| `PAPER_DATA_DIR` | `~/.paper-agent` | 数据库目录 |
| `PAPER_DOWNLOAD_DIR` | （未设） | 对话下载目录；不设时使用显式配置的 `PAPER_DATA_DIR`，否则用已索引论文库目录 |
| `PAPER_NOTE_DIR` | `PAPER_DATA_DIR/notes` | 笔记保存目录（`save_note` 工具与 `/export` 命令） |

配置文件查找顺序为：`PAPER_ENV_FILE`、当前工作目录的 `.env`、editable 源码仓库根目录的 `.env`。显式指定但文件不存在时不会回退到其他 `.env`；非空系统环境变量始终优先。普通 wheel 安装后建议在运行目录放置 `.env`，或设置 `PAPER_ENV_FILE`。

> 首次 `paper index` 需联网下载嵌入模型（~100MB，缓存于本地）。国内网络直连 HuggingFace 常失败：在 `.env` 中设置 `HF_ENDPOINT=https://hf-mirror.com` 与 `HF_HUB_DISABLE_XET=1`（镜像不支持 Xet 存储，缺一不可）。

## 用法

```powershell
paper index            # 索引当前目录（可加参数指定目录，如 paper index .\papers）
paper list                    # 浏览论文库
paper search "注意力机制"      # 混合检索
paper websearch "llm survey"  # 联网检索 arXiv 论文（英文效果更佳）
paper ask "这篇论文提出了什么方法？"  # 问答（需 API key）
paper ask --web "问题"         # 问答时同时联网检索 arXiv 论文
paper chat                    # TUI 对话：需要 API key；联网/写操作需 /confirm
paper serve                   # 启动 Web 界面 http://127.0.0.1:8000
paper status                  # 库与配置状态
paper --version               # 显示当前版本（版本号与 wheel 元数据同源）
```

Windows 下 CLI 会把标准输出和错误输出配置为 UTF-8，含数学符号等 Unicode 文本的检索结果可以直接输出或重定向到 UTF-8 文件。

`paper serve` 默认只监听本机。使用 `--host 0.0.0.0`、非 loopback Host 或 HTTPS 反向代理时必须设置 `PAPER_WEB_API_KEY`；Web 页面会在首次 API 请求时询问 key，并仅在当前浏览器会话中保存。直接远程监听还必须同时传入 `--ssl-certfile` 与 `--ssl-keyfile`，或让 HTTPS 反向代理转发到 `127.0.0.1`（并正确转发原始 scheme/host）。仅在可信隔离网络中，才可显式添加 `--allow-insecure-http` 使用明文 HTTP。

### 隐私边界

PDF 解析、分块、BM25 和向量嵌入均在本地执行。`paper ask`、`paper chat` 和 `paper index --refine` 会把问题以及命中的论文片段或首页文本发送给 `PAPER_LLM_BASE_URL` 指向的第三方服务；`--web`/`websearch` 会把查询词发送给 arXiv。敏感论文应使用可信的自托管兼容接口，或使用 `--no-llm` 纯检索模式。

联网检索基于 arXiv API（免费、无需 key，遵守 3 秒请求间隔）；Web 界面在检索/问答页勾选「联网（arXiv）」即可。

`paper chat` 为终端对话界面（textual）：模型通过 function calling 使用本地库检索、arXiv 搜索、下载、索引、论文列表、库状态和笔记工具。本地只读检索/列表可自动执行；arXiv 联网搜索、下载、重索引、保存笔记不会立即执行，必须由用户检查工具名和参数后输入 `/confirm`，或输入 `/cancel` 取消。Agent 不能切换论文库根目录；切换目录必须在终端显式运行 `paper index <目录> --force`。对话内还支持 `/help`、`/clear`、`/copy`、`/export`、`/quit`。

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
