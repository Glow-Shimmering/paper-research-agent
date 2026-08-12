"""LLM 客户端（OpenAI 兼容）与元数据提炼。"""
import json
import re
from typing import Any, Optional

from openai import OpenAI


class LLMError(Exception):
    pass


class LLMClient:
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
            raise LLMError("未配置 PAPER_LLM_API_KEY")
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

    def chat_with_tools(self, system: str, messages: list[dict], tools: list[dict]) -> dict:
        """带工具调用的一次请求。

        messages 为 OpenAI 格式历史（不含 system）；返回
        {"content": str|None, "tool_calls": [{"id", "name", "arguments": dict}],
        "metadata": {"usage", "finish_reason", "response_id"}}。
        """
        if not self.is_configured:
            raise LLMError("未配置 PAPER_LLM_API_KEY")
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
