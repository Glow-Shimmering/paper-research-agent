# paper-agent

## 项目说明

论文整理与检索 Agent：整理本地论文资料库，提供检索与问答能力。

- 索引本地 PDF 论文库（增量：重复运行只处理变化的文件）
- 混合检索：BM25 关键词 + 本地语义向量（fastembed，CPU 推理），RRF 融合
- 问答：OpenAI 兼容 API（DeepSeek / OpenAI / 通义等），未配置 key 时退回纯检索
- CLI 与本地 Web 界面双入口
- 只读整理：解析元数据建库浏览，不改动你的 PDF 文件

## 安装

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
```

## 配置

复制 `.env.example` 为 `.env` 并填写：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `PAPER_LLM_BASE_URL` | `https://api.deepseek.com` | OpenAI 兼容 API 地址 |
| `PAPER_LLM_API_KEY` | （空） | 留空则问答退回纯检索 |
| `PAPER_LLM_MODEL` | `deepseek-chat` | 问答模型名 |
| `PAPER_EMBED_MODEL` | `BAAI/bge-small-zh-v1.5` | 嵌入模型（首次索引联网下载 ~100MB） |
| `PAPER_DATA_DIR` | `~/.paper-agent` | 数据库目录 |

> 首次 `paper index` 需联网下载嵌入模型（~100MB，缓存于本地）。国内网络直连 HuggingFace 常失败：在 `.env` 中设置 `HF_ENDPOINT=https://hf-mirror.com` 与 `HF_HUB_DISABLE_XET=1`（镜像不支持 Xet 存储，缺一不可）。

## 用法

```powershell
paper index            # 索引当前目录（可加参数指定目录，如 paper index .\papers）
paper list                    # 浏览论文库
paper search "注意力机制"      # 混合检索
paper ask "这篇论文提出了什么方法？"  # 问答（需 API key）
paper serve                   # 启动 Web 界面 http://127.0.0.1:8000
paper status                  # 库与配置状态
```

## 构建与测试

```powershell
.venv\Scripts\python -m pytest -q
```
