# PRAgent 核心链路图

本页只画第一阶段需要负责的范围。虚线表示仅在特定命令或首次运行时发生。

```mermaid
flowchart LR
    user([用户]) --> entry[CLI 或 Web API]
    entry --> mode{运行模式}

    mode -->|"index"| pdf[/本地 PDF/]
    pdf --> parse[PyMuPDF 解析]
    parse --> chunk[按页和段落分块]
    chunk --> embed[FastEmbed 本地向量]
    embed --> sqlite[(SQLite)]

    mode -->|"search/ask"| query[用户问题]
    query --> bm25[BM25 召回]
    query --> vector[向量召回]
    sqlite --> bm25
    sqlite --> vector
    bm25 --> rrf[RRF 融合与 top-k]
    vector --> rrf
    rrf --> llm{配置 LLM?}
    llm -->|"否"| hits[返回检索片段]
    llm -->|"是"| prompt[拼接问题与证据]
    prompt -.-> provider[OpenAI-compatible API]
    provider -.-> verify[引用范围校验]
    verify --> answer[返回答案与来源]

    mode -->|"chat"| chat[chat_turn 循环]
    chat -.-> provider
    provider -.-> call{包含 tool_calls?}
    call -->|"是"| schema[ToolSpec 参数校验]
    schema --> localTool[本地只读工具]
    localTool --> chat
    call -->|"否"| verify

    modelDownload[Hugging Face 模型文件] -.-> embed
```

## 文件映射

| 链路节点 | 主要文件 | 需要回答的问题 |
|---|---|---|
| PDF 解析/分块 | `pdf.py`、`chunking.py` | 页码如何保留？chunk 为什么重叠？ |
| 索引 | `indexer.py`、`embeddings.py` | 哪些 PDF 会跳过？何时重建向量？ |
| 混合检索 | `bm25.py`、`search.py` | 两路候选怎样用 RRF 合并？ |
| 传统问答 | `answer.py`、`llm.py` | 未配置 key 如何降级？`[n]` 怎样校验？ |
| Agent 工具循环 | `chat.py` | assistant/tool 协议消息如何配对？ |
| 工具边界 | `tool_protocol.py`、`tools.py` | 参数、effects、确认和错误怎样约束？ |
| 状态与预算 | `agent.py` | 固定 Planner 与模型选择工具有何区别？ |

## 必须能主动说明的边界

- embedding 在本地执行，但首次模型下载需要联网。
- `ask/chat` 会把问题和检索片段发送给配置的第三方 LLM。
- 引用校验确认 evidence ID 的范围与新鲜度，不自动证明回答和证据语义蕴含。
- `agent.py` 的 Planner 是固定步骤模板；真正的工具选择发生在 `chat.py` 的模型循环中。
