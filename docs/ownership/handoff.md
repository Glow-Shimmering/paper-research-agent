# 接力说明：`codex/ownership-plan`

更新时间：2026-08-27。

本分支用于把 AI 辅助生成的 PRAgent 原型转化为可解释、可测试、可评测的个人作品。总体范围、阅读顺序和六周验收标准见 [接管指南](README.md)，可复现命令见 [第 1 周运行基线](week-01-baseline.md)，调用关系见 [核心链路图](core-flow.md)。

## 已完成

1. `dfd6bdd docs: establish ownership baseline and core flow`
   - 建立干净 Python 3.11 环境并完整运行测试。
   - 使用仓库内三篇论文验证 `index/list/search/ask --no-llm`。
   - 增加运行手册、核心链路图和所有权边界。
   - 显式增加 `httpx[socks]`，修复 SOCKS 代理环境下 embedding 模型下载失败。
2. `4affc5f fix: log and sanitize tool handler failures`
   - 修复 `tools.py::_run_handler` 异常分支引用未定义变量的问题。
   - 对模型返回稳定、脱敏的 `tool_execution_failed`。
   - 本地日志记录 `session_id/run_id/tool_name/error_type`。
   - 增加异常路径测试；该提交后完整测试为 `325 passed`。

## 当前 WIP

`src/pragent/store.py` 已开始增加以下 transcript 存储方法：

- `load_agent_messages(session_id)`；
- `save_agent_messages(session_id, messages)`；
- `delete_agent_session(session_id)`。

这部分只完成了 Store 草稿，尚未接入 Web Agent，也没有针对新增方法的测试。它必须继续按 WIP 对待，不能在简历或 README 中宣称“会话已持久化”。

设计约束：`agent_messages.content` 暂时保存整条 OpenAI-compatible message JSON，以保留 assistant `tool_calls` 和 tool `tool_call_id`。只持久化已闭合的完整回合；等待确认的未决 tool 协议不能作为可恢复上下文写入。

## 下一台电脑的继续顺序

1. 为三个 Store 方法补测试：JSON 无损往返、A/B 会话隔离、第二个 Store 实例重启恢复、原子替换、删除级联、非法 role/session ID。
2. 在 `_SessionRegistry.get()` 创建内存会话时加载已持久化消息。
3. 在正常回合和确认/取消完成后保存消息；存在 `pending_action` 时不保存未闭合回合。
4. 增加幂等清空会话接口；运行中或等待确认时返回 `409`，不能留下仍可执行的孤立 ticket。
5. 增加 Web 测试：会话 A/B 隔离、创建新 app 后恢复、清空后不恢复、同 session 并发拒绝。
6. 补充 history、模型上下文、持久 transcript 和 LangChain4j `ChatMemoryStore` 的对照说明。
7. 完成会话提交后再进入评测脚手架：30 个问题、BM25/vector/RRF、Recall@5/MRR、引用和延迟指标。

## 在新电脑恢复

```bash
git clone https://github.com/Glow-Shimmering/paper-research-agent.git
cd paper-research-agent
git switch codex/ownership-plan
python3.11 -m venv .venv
.venv/bin/python -m pip install -c requirements-dev.lock -e ".[dev]"
.venv/bin/python -m pytest -q
```

真实索引请按 `week-01-baseline.md` 创建专用 `sample_papers/`；不要使用 `pra index .`，否则可能扫描仓库内 pytest 临时 PDF。离线测试不需要 LLM key，真实 `ask/chat` smoke 才需要在本机 `.env` 配置，密钥不得提交。

## 推送与合并边界

- 远程工作分支：`origin/codex/ownership-plan`。
- 本分支尚未合并 `master`。
- 会话持久化 WIP 完成测试前，不要开面向成品的 PR。
- 后续每个阶段保持独立提交：会话持久化、评测脚手架、评测驱动检索调整。
