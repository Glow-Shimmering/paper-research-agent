"""Bounded JSON HTTP transport with cache, throttling, and explicit retry policy."""

from __future__ import annotations

import base64
import email.utils
import json
import os
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from .base import SourceProviderError

_DEFAULT_MAX_RESPONSE_BYTES = 5 * 1024 * 1024
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


@dataclass(frozen=True)
class HttpResponse:
    status: int
    headers: Mapping[str, str]
    body: bytes


class ResponseCache:
    """Content-addressed provider response cache; request headers are never stored."""

    def __init__(
        self,
        directory: str | Path,
        *,
        ttl_seconds: float = 24 * 60 * 60,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        if ttl_seconds < 0:
            raise ValueError("cache ttl_seconds 不能为负数")
        self.directory = Path(directory).expanduser().resolve(strict=False)
        self.ttl_seconds = ttl_seconds
        self._wall_clock = wall_clock
        self._lock = threading.RLock()

    def get(self, provider: str, url: str) -> Optional[HttpResponse]:
        path = self._path(provider, url)
        with self._lock:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                stored_at = float(payload["stored_at"])
                if self._wall_clock() - stored_at > self.ttl_seconds:
                    return None
                body = base64.b64decode(payload["body_b64"], validate=True)
                return HttpResponse(
                    status=int(payload["status"]),
                    headers={str(k): str(v) for k, v in payload["headers"].items()},
                    body=body,
                )
            except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
                return None

    def put(self, provider: str, url: str, response: HttpResponse) -> None:
        if response.status != 200:
            return
        path = self._path(provider, url)
        payload = {
            "version": 1,
            "provider": provider,
            "url": url,
            "status": response.status,
            "headers": {
                key.lower(): value
                for key, value in response.headers.items()
                if key.lower() in {"content-type", "etag", "last-modified"}
            },
            "body_b64": base64.b64encode(response.body).decode("ascii"),
            "stored_at": self._wall_clock(),
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary_name = tempfile.mkstemp(
                prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
            )
            temporary = Path(temporary_name)
            try:
                with os.fdopen(fd, "wb") as file:
                    file.write(encoded)
                    file.flush()
                    os.fsync(file.fileno())
                temporary.chmod(0o600)
                os.replace(temporary, path)
            except Exception:
                temporary.unlink(missing_ok=True)
                raise

    def _path(self, provider: str, url: str) -> Path:
        import hashlib

        digest = hashlib.sha256(f"{provider}\0{url}".encode("utf-8")).hexdigest()
        safe_provider = "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in provider.lower()
        )
        return self.directory / safe_provider / f"{digest}.json"


class RateLimiter:
    def __init__(
        self,
        min_interval: float,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if min_interval < 0:
            raise ValueError("min_interval 不能为负数")
        self.min_interval = float(min_interval)
        self._monotonic = monotonic
        self._sleep = sleep
        self._next_allowed = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = self._monotonic()
            delay = self._next_allowed - now
            if delay > 0:
                self._sleep(delay)
                now = self._monotonic()
            self._next_allowed = max(now, self._next_allowed) + self.min_interval


class JsonHttpClient:
    """GET JSON with bounded retries; cache hits do not consume rate-limit budget."""

    def __init__(
        self,
        provider: str,
        *,
        requester: Optional[Callable[[str, Mapping[str, str], float, int], HttpResponse]] = None,
        cache: Optional[ResponseCache] = None,
        limiter: Optional[RateLimiter] = None,
        timeout: float = 20.0,
        max_response_bytes: int = _DEFAULT_MAX_RESPONSE_BYTES,
        max_retries: int = 2,
        backoff_base: float = 1.0,
        max_retry_delay: float = 30.0,
        sleep: Callable[[float], None] = time.sleep,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        if not provider.strip():
            raise ValueError("provider 不能为空")
        if timeout <= 0 or max_response_bytes <= 0:
            raise ValueError("timeout 和 max_response_bytes 必须大于 0")
        if max_retries < 0 or backoff_base < 0 or max_retry_delay < 0:
            raise ValueError("retry 配置不能为负数")
        self.provider = provider.strip().lower()
        self.requester = requester or _request
        self.cache = cache
        self.limiter = limiter or RateLimiter(0)
        self.timeout = float(timeout)
        self.max_response_bytes = int(max_response_bytes)
        self.max_retries = int(max_retries)
        self.backoff_base = float(backoff_base)
        self.max_retry_delay = float(max_retry_delay)
        self._sleep = sleep
        self._wall_clock = wall_clock

    def get_json(self, url: str, *, headers: Optional[Mapping[str, str]] = None) -> Any:
        if self.cache is not None:
            cached = self.cache.get(self.provider, url)
            if cached is not None:
                return self._decode(cached)

        safe_headers = {str(key): str(value) for key, value in (headers or {}).items()}
        response: Optional[HttpResponse] = None
        for attempt in range(self.max_retries + 1):
            self.limiter.wait()
            try:
                response = self.requester(
                    url,
                    safe_headers,
                    self.timeout,
                    self.max_response_bytes,
                )
            except SourceProviderError as exc:
                if exc.provider == self.provider:
                    raise
                raise SourceProviderError(
                    str(exc),
                    provider=self.provider,
                    code=exc.code,
                    retryable=exc.retryable,
                    status_code=exc.status_code,
                ) from exc
            except Exception as exc:
                if attempt >= self.max_retries:
                    raise SourceProviderError(
                        f"{self.provider} 请求失败：{exc}",
                        provider=self.provider,
                        code="network_error",
                        retryable=True,
                    ) from exc
                self._sleep(min(self.max_retry_delay, self.backoff_base * (2**attempt)))
                continue

            if response.status == 200:
                if self.cache is not None:
                    self.cache.put(self.provider, url, response)
                return self._decode(response)
            if response.status not in _RETRYABLE_STATUS or attempt >= self.max_retries:
                retryable = response.status in _RETRYABLE_STATUS
                raise SourceProviderError(
                    f"{self.provider} HTTP {response.status}",
                    provider=self.provider,
                    code=f"http_{response.status}",
                    retryable=retryable,
                    status_code=response.status,
                )
            delay = _retry_delay(
                response.headers.get("Retry-After")
                or response.headers.get("retry-after"),
                default=self.backoff_base * (2**attempt),
                now=self._wall_clock(),
            )
            self._sleep(min(self.max_retry_delay, max(0.0, delay)))

        raise AssertionError("unreachable provider retry loop")

    def _decode(self, response: HttpResponse) -> Any:
        if len(response.body) > self.max_response_bytes:
            raise SourceProviderError(
                f"{self.provider} 响应超过 {self.max_response_bytes} 字节限制",
                provider=self.provider,
                code="response_too_large",
            )
        try:
            return json.loads(response.body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SourceProviderError(
                f"{self.provider} 返回无效 JSON",
                provider=self.provider,
                code="invalid_json",
            ) from exc


def _request(
    url: str,
    headers: Mapping[str, str],
    timeout: float,
    max_response_bytes: int,
) -> HttpResponse:
    # 复用 SSRF-safe 单跳请求：协议/凭据/私网地址校验并固定解析 IP；
    # 非法目标、超限响应与网络故障显式失败，由 JsonHttpClient 统一重试。
    from ..ingestion.safe_fetch import SafeFetchError, pinned_get

    try:
        response = pinned_get(
            url,
            headers=dict(headers),
            timeout=timeout,
            max_bytes=max_response_bytes,
        )
    except SafeFetchError as exc:
        raise SourceProviderError(
            str(exc),
            provider="http",
            code=exc.code,
            retryable=exc.retryable,
            status_code=exc.status_code,
        ) from exc
    response_headers = {str(key): str(value) for key, value in response.headers.items()}
    return HttpResponse(int(response.status), response_headers, response.body)


def _retry_delay(value: Optional[str], *, default: float, now: float) -> float:
    if value:
        try:
            return max(0.0, float(value.strip()))
        except ValueError:
            try:
                parsed = email.utils.parsedate_to_datetime(value)
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                return max(0.0, parsed.timestamp() - now)
            except (TypeError, ValueError, OverflowError):
                pass
    return max(0.0, default)
