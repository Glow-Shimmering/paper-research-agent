# Web Agent 会话记忆边界

这份说明用于避免把浏览器展示、模型上下文、数据库 transcript 和框架抽象混为一谈。

| 概念 | PRAgent 中的含义 | 是否跨重启 | 主要边界 |
|---|---|---:|---|
| 页面 history | 服务端从 transcript 投影的用户消息、回答和工具卡片 | 是 | 刷新后由 session state API 重绘；不是独立事实源 |
| 模型上下文 | 本轮传给 `llm.chat_with_tools()` 的 OpenAI-compatible messages | 间接支持 | 来自内存热会话；首次访问某 session 时从 transcript 恢复 |
| 持久 transcript | `agent_sessions` + `agent_messages` 中按 `seq` 保存的完整 message JSON | 是 | 保存 assistant `tool_calls` 与 tool `tool_call_id`，供模型协议无损恢复 |
| 待确认票据 | `pending_actions` 中冻结的工具名、参数、摘要、tool call 与 run | 是 | 恢复时校验两类摘要；确认前 CAS 认领，只能继续或取消原动作 |
| run/event 审计 | `agent_runs` 与 `agent_events` 中的状态、预算和摘要事件 | 是 | 用于审计，不等同于可直接回填模型的对话消息 |
| LangChain4j `ChatMemoryStore` | 框架定义的按 memory ID 读取、更新、删除消息的存储接口 | 取决于实现 | PRAgent 未引入 LangChain4j；Store 三个方法承担相似职责，但协议和生命周期由本项目控制 |

## 未闭合 transcript 如何安全保存

OpenAI-compatible 工具协议要求 assistant 的每个 `tool_call_id` 后面出现对应 tool message。等待用户确认时，assistant tool call 仍未闭合；此时把消息写成可恢复上下文，会让重启后的模型收到不完整协议，或者让已经失去冻结票据的动作看起来仍可确认。

因此当前边界是把未闭合消息和恢复所需票据放在同一事务中，而不是只保存其中一半：

- 正常完成的回合保存；
- 用户确认并执行工具后，补齐原始 `tool_call_id` 再保存；
- 用户取消后，写入“未执行”的 tool result 闭合协议再保存；
- 存在 `pending_action` 时，同时保存 assistant tool call、冻结参数、参数 SHA-256 与 action digest；
- 运行中或等待确认时拒绝清空，避免留下仍可执行的孤立动作；
- 服务在等待确认期间重启时，复核票据绑定后恢复同一确认卡；不会自动执行；
- 确认前把票据从 `pending` CAS 为 `approved`，只有认领成功的请求可以执行工具。

## 当前调用关系

```text
POST /api/agent/chat
  -> _SessionRegistry.get(session_id, project_id)
  -> Store.load_agent_session_state(session_id)
  -> 追加本轮 user message
  -> chat_turn(...)
  -> Store.save_agent_session_state(messages, pending_action?)

GET /api/agent/sessions/{session_id}
  -> 恢复 project/history/run/pending 卡片

POST /api/agent/confirm
  -> Store.claim_pending_action(action_id)  # pending -> approved CAS
  -> 仅执行冻结参数
  -> executed/rejected 终态 + 闭合 transcript

DELETE /api/agent/sessions/{session_id}
  -> 运行中或待确认：409
  -> 空闲：删除 agent_sessions，级联删除 agent_messages/pending_actions
```
