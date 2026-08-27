# PRAgent 核心链路接管指南

这组文档服务于“把 AI 辅助生成的原型转化为本人能解释、能修改、能验证的作品”。目标不是短期读完全部源码，而是接管一条最小纵向链路：

```text
PDF -> 解析/分块 -> BM25/向量检索 -> RRF -> LLM -> 引用校验 -> CLI/Web
```

Agent 部分只增加一条受控工具链：

```text
用户消息 -> 模型 tool_calls -> ToolSpec 校验 -> 本地只读工具
         -> tool result 回填 -> 最终引用校验
```

## 当前可信基线

- 基线日期：2026-08-26。
- 环境：macOS arm64、CPython 3.11.15、`requirements-dev.lock`。
- 完整离线测试：`324 passed`；第三方依赖弃用警告的数量可能随缓存和解释器补丁版本变化。
- 样本库：仓库根目录的 3 篇推荐系统论文，共索引 259 个分块。
- 已验证命令：`status`、`index`、`list`、`search`、`ask --no-llm`。
- 未验证：真实 LLM 的 `ask` 与 `chat`。它们必须使用本人配置的 OpenAI-compatible provider 单独做 live smoke，不能由脚本化测试替代。

详细复现步骤见 [第 1 周运行基线](week-01-baseline.md)，核心调用关系见 [核心链路图](core-flow.md)。跨电脑继续开发时先读[接力说明](handoff.md)。

## 阅读顺序与所有权边界

1. `answer.py`、`search.py`：问答与混合检索。
2. `indexer.py`、`pdf.py`、`chunking.py`、`embeddings.py`：索引管线。
3. `llm.py`：OpenAI-compatible 模型边界。
4. `chat.py`：模型与工具的循环。
5. `tool_protocol.py` 与 `tools.py` 中一个本地只读 handler。
6. `agent.py`、`agent_eval.py`：状态、预算和确定性场景。

第一阶段暂不承担：多 provider 聚合、SSRF 抓取、后台 job lease、完整迁移历史、双 Web 前端、Textual TUI、九栏 Deep Read 和全部工具。代码可以保留，但面试时不把这些模块描述为个人已掌握贡献。

## 每周必须留下的证据

- 一份本人用自己的语言写的调用链或设计说明。
- 一个能够复跑的命令、测试或评测结果。
- 一个真实失败及定位过程，不能只记录成功截图。
- 涉及代码的修改使用独立提交，提交说明包含现象、根因、改动和验证。

## 进入简历前的验收

- 不看源码，5 分钟画出 RAG 与单工具调用链。
- 从干净环境启动并运行核心测试和一次 live smoke。
- 解释 history、模型上下文和持久化 transcript 的差别。
- 至少完成异常处理、会话持久化、评测驱动检索调整三个本人负责的提交。
- 报告 Recall@5/MRR、引用合法率、人工证据支持率、延迟和 token，而不是只报告代码量。
- 坦诚说明早期原型使用 AI 辅助，后续范围收敛、审查、修复、测试与评测由本人负责。
