"""LLM 对话循环：模型可调用工具，执行结果回传，直至给出最终回答。"""
import json
from dataclasses import dataclass
from typing import Optional

from .tools import TOOLS as TOOL_SCHEMAS
from .tools import ToolContext, execute_tool

MAX_TOOL_ROUNDS = 10

SYSTEM_PROMPT = (
    "你是一个论文研究助手，帮助用户整理、检索和分析论文。"
    "你可以调用工具：本地论文库检索（local_search）、arXiv 联网搜索（web_search）、"
    "下载论文到本地库（download_paper）、索引论文目录（index_papers）、"
    "列出库中论文（list_papers）、查看库状态（library_status）。"
    "需要信息时先调用工具获取事实，再基于事实回答；不要编造工具结果。"
    "回答用中文（除非用户使用其他语言）。"
)


@dataclass
class TurnLog:
    """单次对话的展示日志（供 UI 渲染）。"""

    role: str  # user | assistant | tool | error
    content: str = ""
    tool_name: Optional[str] = None
    tool_args: Optional[dict] = None
    tool_result: Optional[str] = None


def chat_turn(llm, messages: list[dict], ctx: ToolContext) -> tuple[list[dict], list[TurnLog]]:
    """执行一轮对话（含最多 MAX_TOOL_ROUNDS 轮工具调用）。

    messages 为 OpenAI 格式历史（不含 system），原地扩展并返回；
    logs 为按序的展示日志。
    """
    logs: list[TurnLog] = []
    for _ in range(MAX_TOOL_ROUNDS):
        resp = llm.chat_with_tools(SYSTEM_PROMPT, messages, TOOL_SCHEMAS)
        content = resp["content"]
        tool_calls = resp["tool_calls"]

        messages.append(
            {
                "role": "assistant",
                "content": content,
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {"name": tc["name"], "arguments": _json_dumps(tc["arguments"])},
                    }
                    for tc in tool_calls
                ],
            }
        )
        logs.append(TurnLog(role="assistant", content=content or ""))

        if not tool_calls:
            break

        for tc in tool_calls:
            try:
                result = execute_tool(tc["name"], tc["arguments"], ctx)
            except Exception as exc:  # 工具实现内部异常兜底
                result = f"工具执行失败：{exc}"
            logs.append(
                TurnLog(
                    role="tool",
                    tool_name=tc["name"],
                    tool_args=tc["arguments"],
                    tool_result=result,
                )
            )
            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
    else:
        logs.append(TurnLog(role="error", content="工具调用轮次过多，已停止。"))
    return messages, logs


def _json_dumps(obj: dict) -> str:
    return json.dumps(obj, ensure_ascii=False)
