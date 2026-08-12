"""结构化工具协议：分类、参数校验、结果与审批票据。"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping, Optional


class ToolEffect(str, Enum):
    """工具可观察副作用；每个工具注册时必须显式声明。"""

    READ_LOCAL = "read_local"
    WRITE_LOCAL = "write_local"
    NETWORK = "network"


class ToolValidationError(ValueError):
    """工具定义或调用参数不满足协议。"""


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: Mapping[str, Any]
    handler: Callable[..., Any] = field(repr=False, compare=False)
    effects: frozenset[ToolEffect]
    timeout_seconds: float = 30.0
    idempotent: bool = True

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.name, str):
            raise ToolValidationError("工具 name 必须是非空字符串")
        if not callable(self.handler):
            raise ToolValidationError(f"工具 {self.name} handler 不可调用")
        if self.effects is None:
            raise ToolValidationError(f"工具 {self.name} 未声明 effects")
        try:
            normalized_effects = frozenset(self.effects)
        except TypeError as exc:
            raise ToolValidationError(f"工具 {self.name} effects 非法") from exc
        if not normalized_effects:
            raise ToolValidationError(f"工具 {self.name} effects 不能为空")
        if any(not isinstance(effect, ToolEffect) for effect in normalized_effects):
            raise ToolValidationError(f"工具 {self.name} 包含未分类 effect")
        object.__setattr__(self, "effects", normalized_effects)
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not math.isfinite(float(self.timeout_seconds))
            or self.timeout_seconds <= 0
        ):
            raise ToolValidationError(f"工具 {self.name} timeout_seconds 必须是正数")
        if not isinstance(self.idempotent, bool):
            raise ToolValidationError(f"工具 {self.name} idempotent 必须是布尔值")
        if self.parameters.get("type") != "object":
            raise ToolValidationError(f"工具 {self.name} parameters 必须是 object schema")
        properties = self.parameters.get("properties")
        if not isinstance(properties, Mapping):
            raise ToolValidationError(f"工具 {self.name} properties 必须是对象")
        required = self.parameters.get("required", [])
        if not isinstance(required, list) or any(key not in properties for key in required):
            raise ToolValidationError(f"工具 {self.name} required 引用了未知字段")

    @property
    def mutating(self) -> bool:
        return ToolEffect.WRITE_LOCAL in self.effects

    @property
    def external(self) -> bool:
        return ToolEffect.NETWORK in self.effects

    @property
    def needs_confirmation(self) -> bool:
        return self.mutating or self.external

    def openai_schema(self) -> dict[str, Any]:
        parameters = copy.deepcopy(dict(self.parameters))
        parameters["additionalProperties"] = False
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": parameters,
            },
        }


@dataclass(frozen=True)
class ToolResult:
    """执行器和未来编排器使用的机器可读工具结果。"""

    ok: bool
    code: str
    message: str = ""
    data: Any = None
    evidence_ids: tuple[str, ...] = ()
    retryable: bool = False
    requires_confirmation: bool = False
    action_id: Optional[str] = None
    digest: Optional[str] = None

    @classmethod
    def success(
        cls,
        *,
        data: Any = None,
        message: str = "",
        evidence_ids: tuple[str, ...] = (),
    ) -> "ToolResult":
        return cls(
            ok=True,
            code="ok",
            message=message,
            data=data,
            evidence_ids=evidence_ids,
        )

    @classmethod
    def error(
        cls,
        code: str,
        message: str,
        *,
        data: Any = None,
        evidence_ids: tuple[str, ...] = (),
        retryable: bool = False,
    ) -> "ToolResult":
        return cls(
            ok=False,
            code=code,
            message=message,
            data=data,
            evidence_ids=evidence_ids,
            retryable=retryable,
        )

    def to_text(self) -> str:
        """兼容旧 function-calling 消息所需的字符串表示。"""
        if self.data is None:
            return self.message
        rendered = json.dumps(self.data, ensure_ascii=False, default=str)
        return f"{self.message}\n{rendered}" if self.message else rendered

    def to_model_text(self) -> str:
        """编排器合同名称；旧调用方继续使用 ``to_text``。"""
        return self.to_text()


def canonical_action_digest(
    name: str,
    args: Mapping[str, Any],
    *,
    tool_call_id: Optional[str] = None,
    run_id: Optional[str] = None,
) -> str:
    payload = {
        "name": name,
        "args": args,
        "tool_call_id": tool_call_id,
        "run_id": run_id,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(eq=False)
class PendingAction:
    """可持久化审批票据，同时保持旧二元 tuple 的读取/比较语义。"""

    name: str
    args: dict[str, Any]
    action_id: str
    digest: str
    tool_call_id: Optional[str] = None
    run_id: Optional[str] = None

    @classmethod
    def create(
        cls,
        name: str,
        args: Mapping[str, Any],
        *,
        tool_call_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> "PendingAction":
        frozen_args = copy.deepcopy(dict(args))
        return cls(
            name=name,
            args=frozen_args,
            action_id=f"act_{uuid.uuid4().hex}",
            digest=canonical_action_digest(
                name, frozen_args, tool_call_id=tool_call_id, run_id=run_id
            ),
            tool_call_id=tool_call_id,
            run_id=run_id,
        )

    def is_bound(self) -> bool:
        return self.digest == canonical_action_digest(
            self.name,
            self.args,
            tool_call_id=self.tool_call_id,
            run_id=self.run_id,
        )

    def __iter__(self):
        yield self.name
        yield self.args

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int):
        return (self.name, self.args)[index]

    def __eq__(self, other: object) -> bool:
        if isinstance(other, PendingAction):
            return (
                self.name,
                self.args,
                self.action_id,
                self.digest,
                self.tool_call_id,
                self.run_id,
            ) == (
                other.name,
                other.args,
                other.action_id,
                other.digest,
                other.tool_call_id,
                other.run_id,
            )
        if isinstance(other, tuple) and len(other) == 2:
            return (self.name, self.args) == other
        return False


@dataclass(frozen=True)
class ConfirmedAction:
    name: str
    args: dict[str, Any]
    action_id: str
    digest: str
    result: ToolResult
    tool_call_id: Optional[str] = None
    run_id: Optional[str] = None


def validate_tool_arguments(spec: ToolSpec, args: Any) -> dict[str, Any]:
    """严格校验 object schema 的常用 JSON 子集，不接受未知字段。"""
    if not isinstance(args, dict):
        raise ToolValidationError("参数必须是 JSON object")
    properties = spec.parameters.get("properties", {})
    required = spec.parameters.get("required", [])
    unknown = sorted(set(args) - set(properties))
    if unknown:
        raise ToolValidationError(f"包含未知字段：{', '.join(unknown)}")
    missing = [name for name in required if name not in args]
    if missing:
        raise ToolValidationError(f"缺少必填字段：{', '.join(missing)}")
    validated = copy.deepcopy(args)
    for name, value in validated.items():
        _validate_value(name, value, properties[name])
    return validated


def _validate_value(name: str, value: Any, schema: Mapping[str, Any]) -> None:
    expected = schema.get("type")
    expected_types = expected if isinstance(expected, list) else [expected]
    if not any(_matches_type(value, type_name) for type_name in expected_types):
        labels = "/".join(str(item) for item in expected_types)
        raise ToolValidationError(f"字段 {name} 必须是 {labels}")
    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            raise ToolValidationError(f"字段 {name} 长度不能小于 {schema['minLength']}")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            raise ToolValidationError(f"字段 {name} 长度不能超过 {schema['maxLength']}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            raise ToolValidationError(f"字段 {name} 不能小于 {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            raise ToolValidationError(f"字段 {name} 不能大于 {schema['maximum']}")
    if "enum" in schema and value not in schema["enum"]:
        raise ToolValidationError(f"字段 {name} 不在允许值中")


def _matches_type(value: Any, type_name: Any) -> bool:
    if type_name == "string":
        return isinstance(value, str)
    if type_name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if type_name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if type_name == "boolean":
        return isinstance(value, bool)
    if type_name == "object":
        return isinstance(value, dict)
    if type_name == "array":
        return isinstance(value, list)
    if type_name == "null":
        return value is None
    return False
