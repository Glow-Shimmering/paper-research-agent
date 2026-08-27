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
3. `feat: persist closed web agent transcripts`（本提交）
   - Store transcript 完成 JSON 无损往返、A/B 隔离、重启恢复、原子替换、删除级联与非法输入测试。
   - `_SessionRegistry` 首次创建热会话时加载已持久化消息；正常、确认和取消后的闭合协议会保存。
   - 修复 Web 用户问题只进入 run objective、没有进入模型 messages 的回归。
   - 新增幂等 `DELETE /api/agent/sessions/{session_id}`；运行中或等待确认时返回 `409`。
   - 新会话按钮先安全清空服务端 transcript，再更换浏览器 session ID。
   - 新增 `lxml-html-clean` 直接依赖；无 Windows symlink 权限时只跳过对应安全测试。
   - Windows 验证：`338 passed, 1 skipped`、临时目录 75 MB、Phase 3 offline smoke、`pip check`、Python compileall、`node --check`、wheel build/check 全部通过。
4. `feat: add auditable retrieval evaluation`（本提交）
   - 新增 `pra-core-30-v1`：三篇论文各 10 题，英文/中文各 15 题，固定论文 SHA、chunk 序号和精确原文摘录。
   - 同一路径显式运行 BM25、vector 与 RRF，报告 Recall@5、MRR、预热后延迟、逐题 evidence 和失败明细。
   - 标签在运行前反查当前 SQLite；不存在、摘录漂移或回答 evidence 白名单越界时拒绝评测。
   - 首次真实结果：BM25 `0.6000/0.5067`、vector `0.4333/0.3428`、RRF `0.6667/0.5122`（Recall@5/MRR）。
   - 5/5 示例回答通过引用合法性检查；人工证据支持率保持 `null`，等待真人复核，不宣称语义蕴含。
   - Windows 验证：`345 passed, 1 skipped`、临时目录 76 MB、Phase 3 offline smoke、`pip check`、Python compileall、`node --check`、wheel build/check 全部通过。

## 当前状态

Web Agent 的闭合 transcript 现在可以跨进程恢复：

- `load_agent_messages(session_id)`；
- `save_agent_messages(session_id, messages)`；
- `delete_agent_session(session_id)`。

`agent_messages.content` 保存整条 OpenAI-compatible message JSON，以保留 assistant `tool_calls` 和 tool `tool_call_id`。只持久化已闭合的完整回合；等待确认的未决 tool 协议不会覆盖上一个持久边界。详细概念对照见 [Web Agent 会话记忆边界](session-memory.md)。

仍未完成的产品化范围：页面刷新后重绘历史卡片、session 绑定 project，以及把冻结 pending action 跨重启恢复。这些属于产品路线 Step 25 的剩余部分，不能因 transcript 已完成而一并宣称完成。

## 下一阶段：评测驱动检索调整

1. 先把评测脚手架作为独立提交，确认工作树干净。
2. 冻结 `pra-core-30-v1` 与同一三论文 SQLite 快照，不为提高分数修改标签。
3. 从 RRF 的 10 个失败题目中归纳一个可解释的检索问题并完成一次窄调整。
4. 在同一快照重跑，报告 before/after Recall@5、MRR、延迟和 paired query delta。
5. 将检索调整单独提交；如果指标没有改善，也如实记录并回退该调整，不污染评测基线提交。

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
- 会话 transcript 阶段必须先独立提交，再开始评测脚手架。
- 后续每个阶段保持独立提交：会话持久化、评测脚手架、评测驱动检索调整。
