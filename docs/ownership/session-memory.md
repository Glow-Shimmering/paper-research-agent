# Web Agent 会话记忆边界

这份说明用于避免把浏览器展示、模型上下文、数据库 transcript 和框架抽象混为一谈。

| 概念 | PRAgent 中的含义 | 是否跨重启 | 主要边界 |
|---|---|---:|---|
| 页面 history | 浏览器当前页面已经渲染的用户消息、回答和工具卡片 | 否 | 只是 UI；刷新后当前版本不会重新绘制旧卡片 |
| 模型上下文 | 本轮传给 `llm.chat_with_tools()` 的 OpenAI-compatible messages | 间接支持 | 来自内存热会话；首次访问某 session 时从 transcript 恢复 |
| 持久 transcript | `agent_sessions` + `agent_messages` 中按 `seq` 保存的完整 message JSON | 是 | 保存 assistant `tool_calls` 与 tool `tool_call_id`，供模型协议无损恢复 |
| run/event 审计 | `agent_runs` 与 `agent_events` 中的状态、预算和摘要事件 | 是 | 用于审计，不等同于可直接回填模型的对话消息 |
| LangChain4j `ChatMemoryStore` | 框架定义的按 memory ID 读取、更新、删除消息的存储接口 | 取决于实现 | PRAgent 未引入 LangChain4j；Store 三个方法承担相似职责，但协议和生命周期由本项目控制 |

## 为什么只保存闭合 transcript

OpenAI-compatible 工具协议要求 assistant 的每个 `tool_call_id` 后面出现对应 tool message。等待用户确认时，assistant tool call 仍未闭合；此时把消息写成可恢复上下文，会让重启后的模型收到不完整协议，或者让已经失去冻结票据的动作看起来仍可确认。

因此当前边界是：

- 正常完成的回合保存；
- 用户确认并执行工具后，补齐原始 `tool_call_id` 再保存；
- 用户取消后，写入“未执行”的 tool result 闭合协议再保存；
- 存在内存 `pending_action` 时不覆盖上一个持久化边界；
- 运行中或等待确认时拒绝清空，避免留下仍可执行的孤立动作；
- 服务在等待确认期间重启时，只恢复上一个闭合 transcript，不恢复或重新执行未决动作。

这和产品路线中的“冻结 pending action 跨重启恢复”不是同一能力。后者还需要把确认票据、工具版本、参数摘要、run 状态和幂等边界一起接入 `pending_actions`，本阶段没有宣称完成。

## 当前调用关系

```text
POST /api/agent/chat
  -> _SessionRegistry.get(session_id)
  -> Store.load_agent_messages(session_id)  # 仅首次创建热会话
  -> 追加本轮 user message
  -> chat_turn(...)
  -> 无 pending_action 时 Store.save_agent_messages(...)

DELETE /api/agent/sessions/{session_id}
  -> 运行中或待确认：409
  -> 空闲：删除 agent_sessions，级联删除 agent_messages
```
