"""LLM 客户端（OpenAI 兼容）与元数据提炼。"""
import json
import re
from typing import Any, Callable, Iterator, Optional

from openai import OpenAI


class LLMError(Exception):
    pass


class LLMClient:
    # 声明流式能力；chat_turn 等编排方据此决定是否透传 on_delta（脚本化
    # 测试替身没有该标记，自动保持旧的非流式调用签名）。
    supports_streaming = True

    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 60.0):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self._client: Optional[OpenAI] = None
        self._timeout = timeout
        self.last_response_metadata: dict[str, Any] = {}

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    def _get_client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(
                base_url=self.base_url, api_key=self.api_key, timeout=self._timeout
            )
        return self._client

    def chat(self, system: str, user: str) -> str:
        """生成式对话。失败抛 LLMError 带原因。"""
        return self.chat_with_metadata(system, user)["content"]

    def chat_with_metadata(self, system: str, user: str) -> dict[str, Any]:
        """生成式对话，同时返回可审计的响应元数据。

        ``chat`` 保持原有字符串返回值；需要 usage / finish_reason /
        response_id 的调用方使用本方法或读取 ``last_response_metadata``。
        """
        if not self.is_configured:
            raise LLMError("未配置 PRA_LLM_API_KEY")
        try:
            resp = self._get_client().chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.2,
            )
            content = resp.choices[0].message.content
            if content is None:
                raise LLMError("LLM 返回空内容")
            metadata = _response_metadata(resp)
            self.last_response_metadata = metadata
            return {"content": content, "metadata": metadata}
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(f"LLM 调用失败：{exc}") from exc

    def chat_stream(self, system: str, user: str) -> Iterator[str]:
        """流式生成式对话：逐段产出回答文本。

        迭代结束后 ``last_response_metadata`` 保存 usage / finish_reason /
        response_id（``include_usage`` 使 usage 出现在最后一个块中）。
        迭代中途的异常统一包装为 ``LLMError``。
        """
        if not self.is_configured:
            raise LLMError("未配置 PRA_LLM_API_KEY")
        try:
            stream = self._get_client().chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.2,
                stream=True,
                stream_options={"include_usage": True},
            )
        except Exception as exc:
            raise LLMError(f"LLM 调用失败：{exc}") from exc
        last_chunk = None
        try:
            for chunk in stream:
                last_chunk = chunk
                choices = getattr(chunk, "choices", None)
                if not choices:
                    continue
                delta = getattr(choices[0], "delta", None)
                piece = getattr(delta, "content", None) if delta is not None else None
                if piece:
                    yield piece
        except Exception as exc:
            raise LLMError(f"LLM 调用失败：{exc}") from exc
        self.last_response_metadata = (
            _response_metadata(last_chunk) if last_chunk is not None else {}
        )

    def chat_with_tools(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict],
        *,
        on_delta: Optional[Callable[[str], None]] = None,
    ) -> dict:
        """带工具调用的一次请求。

        messages 为 OpenAI 格式历史（不含 system）；返回
        {"content": str|None, "tool_calls": [{"id", "name", "arguments": dict}],
        "metadata": {"usage", "finish_reason", "response_id"}}。

        传入 ``on_delta`` 时使用流式请求：每个内容增量立即回调（供 TUI/Web
        逐字渲染最终回答），工具调用片段按 index 重组；usage 从最后一个块
        读取，行为与非流式一致。带 tool_calls 的响应通常没有内容增量。
        """
        if not self.is_configured:
            raise LLMError("未配置 PRA_LLM_API_KEY")
        if on_delta is None:
            return self._chat_with_tools_plain(system, messages, tools)
        try:
            stream = self._get_client().chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system}, *messages],
                tools=tools,
                tool_choice="auto",
                temperature=0.2,
                stream=True,
                stream_options={"include_usage": True},
            )
        except Exception as exc:
            raise LLMError(f"LLM 调用失败：{exc}") from exc
        content_parts: list[str] = []
        tool_fragments: dict[int, dict[str, str]] = {}
        last_chunk = None
        try:
            for chunk in stream:
                last_chunk = chunk
                choices = getattr(chunk, "choices", None)
                if not choices:
                    continue
                delta = getattr(choices[0], "delta", None)
                if delta is None:
                    continue
                piece = getattr(delta, "content", None)
                if piece:
                    content_parts.append(piece)
                    on_delta(piece)
                for frag in getattr(delta, "tool_calls", None) or []:
                    index = int(getattr(frag, "index", 0))
                    slot = tool_fragments.setdefault(
                        index, {"id": "", "name": "", "arguments": ""}
                    )
                    if getattr(frag, "id", None):
                        slot["id"] = frag.id
                    function = getattr(frag, "function", None)
                    if function is None:
                        continue
                    if getattr(function, "name", None):
                        slot["name"] += function.name
                    if getattr(function, "arguments", None):
                        slot["arguments"] += function.arguments
        except Exception as exc:
            raise LLMError(f"LLM 调用失败：{exc}") from exc
        metadata = _response_metadata(last_chunk) if last_chunk is not None else {}
        tool_calls = []
        for index in sorted(tool_fragments):
            slot = tool_fragments[index]
            try:
                args = json.loads(slot["arguments"]) if slot["arguments"] else {}
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(
                {"id": slot["id"], "name": slot["name"], "arguments": args}
            )
        content = "".join(content_parts) or None
        self.last_response_metadata = metadata
        return {"content": content, "tool_calls": tool_calls, "metadata": metadata}

    def _chat_with_tools_plain(
        self, system: str, messages: list[dict], tools: list[dict]
    ) -> dict:
        try:
            resp = self._get_client().chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system}, *messages],
                tools=tools,
                tool_choice="auto",
                temperature=0.2,
            )
        except Exception as exc:
            raise LLMError(f"LLM 调用失败：{exc}") from exc
        message = resp.choices[0].message
        metadata = _response_metadata(resp)
        self.last_response_metadata = metadata
        tool_calls = []
        for tc in message.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}
            tool_calls.append({"id": tc.id, "name": tc.function.name, "arguments": args})
        return {"content": message.content, "tool_calls": tool_calls, "metadata": metadata}


def _response_metadata(response: Any) -> dict[str, Any]:
    """从 OpenAI 兼容响应提炼 JSON 可序列化元数据。"""
    choices = getattr(response, "choices", None) or []
    finish_reason = getattr(choices[0], "finish_reason", None) if choices else None
    usage_obj = getattr(response, "usage", None)
    usage: dict[str, Any] = {}
    if usage_obj is not None:
        if hasattr(usage_obj, "model_dump"):
            dumped = usage_obj.model_dump(exclude_none=True)
            if isinstance(dumped, dict):
                usage = dumped
        elif isinstance(usage_obj, dict):
            usage = dict(usage_obj)
        else:
            for key in (
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "input_tokens",
                "output_tokens",
            ):
                value = getattr(usage_obj, key, None)
                if value is not None:
                    usage[key] = value
    return {
        "usage": usage,
        "finish_reason": finish_reason,
        "response_id": getattr(response, "id", None),
    }


_REFINE_SYSTEM = (
    "你是论文元数据提取器。根据论文首页文本提取标题、作者列表和发表年份，"
    '只输出 JSON：{"title": "...", "authors": ["..."], "year": 2023}。无法确定的字段用 null。'
)


def refine_metadata(llm, filename: str, first_text: str) -> Optional[dict]:
    """LLM 提炼元数据；解析失败或字段非法返回 None（调用方保留原值）。"""
    try:
        raw = llm.chat(_REFINE_SYSTEM, f"文件名：{filename}\n\n首页文本：\n{first_text}")
    except LLMError:
        return None
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    result: dict = {}
    title = data.get("title")
    if isinstance(title, str) and title.strip():
        result["title"] = title.strip()
    authors = data.get("authors")
    if isinstance(authors, list):
        cleaned = [a.strip() for a in authors if isinstance(a, str) and a.strip()]
        if cleaned:
            result["authors"] = cleaned
    year = data.get("year")
    if isinstance(year, int) and 1900 <= year <= 2100:
        result["year"] = year
    return result
