# 第 1 周：可复现运行基线

## 1. 创建干净环境

macOS/Linux：

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -c requirements-dev.lock -e ".[dev]"
.venv/bin/python -m pip check
.venv/bin/python -m pytest -q
```

Windows PowerShell：

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -c requirements-dev.lock -e ".[dev]"
.venv\Scripts\python -m pip check
.venv\Scripts\python -m pytest -q
```

预期基线为 `324 passed`。第三方依赖的弃用警告应单独记录，但不能当作项目测试失败。

## 2. 建立专用样本目录

不要运行 `pra index .`：pytest 的 `--basetemp=.pytest-tmp` 位于仓库内，直接扫描仓库根目录可能把测试生成的 PDF fixture 一并索引。

```bash
mkdir -p sample_papers
cp 2604.05113.pdf 2604.18351.pdf 2605.18805.pdf sample_papers/
```

`sample_papers/` 已被 `.gitignore` 忽略，不会污染提交。

## 3. 使用隔离数据目录跑通离线闭环

先在当前 shell 指定一个学习用数据目录：

```bash
export PRA_DATA_DIR="$PWD/.local/pra-data"
.venv/bin/pra status
.venv/bin/pra index sample_papers
.venv/bin/pra list
.venv/bin/pra search "large language model" --top 5 --json
.venv/bin/pra ask "这些论文主要研究什么？" --no-llm --top 5 --json
```

基线样本应得到 3 篇论文、259 个分块。分块数受 PDF 内容和 chunking 逻辑影响；代码改变后如有差异，应解释原因而不是机械修改数字。

## 4. 真实模型 smoke

复制 `.env.example` 为 `.env`，只在本地填写以下变量，禁止提交密钥：

```dotenv
PRA_LLM_BASE_URL=https://api.deepseek.com
PRA_LLM_API_KEY=
PRA_LLM_MODEL=deepseek-chat
```

配置后运行：

```bash
.venv/bin/pra ask "RecoAtlas 如何评价推荐集合？" --top 8 --no-stream
.venv/bin/pra chat
```

`chat` 至少人工验证一次 `library_status` 或 `list_papers` 本地只读工具调用。记录 provider、模型、时间、工具名、最终状态和引用；不要记录 API key 或完整敏感论文文本。

## 5. 已知失败与解释

| 现象 | 根因 | 处理 |
|---|---|---|
| `pra index .` 扫描数远大于 3 | 仓库内存在 pytest 临时 PDF | 只索引 `sample_papers` |
| SOCKS proxy 提示缺少 `socksio` | `httpx` 的 SOCKS 支持是 optional extra | 项目显式依赖 `httpx[socks]` |
| `ask` 返回 `retrieval_only=true` | 未配置 `PRA_LLM_API_KEY` | 这是预期降级；另做 live smoke |
| 首次索引较慢 | 需要下载并初始化 BGE embedding 模型 | 保留缓存；记录模型名与首次耗时 |

## 6. 本周本人验收

不看本文，口述并重新执行：创建环境、运行测试、索引三篇论文、查看库状态、完成一次检索。若仍需复制全部命令，继续本周任务，不进入下一阶段。
