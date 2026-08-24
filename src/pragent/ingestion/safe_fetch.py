"""SSRF-safe, redirect-aware HTML fetch with DNS pinning and hard body limits."""

from __future__ import annotations

import http.client
import ipaddress
import socket
import ssl
import time
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Optional, Protocol
from urllib.parse import quote, urljoin, urlsplit, urlunsplit

_ALLOWED_MIME_TYPES = frozenset({"text/html", "application/xhtml+xml"})
_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})


class SafeFetchError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        code: str,
        retryable: bool = False,
        status_code: Optional[int] = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable
        self.status_code = status_code


@dataclass(frozen=True)
class FetchPolicy:
    max_redirects: int = 5
    timeout_seconds: float = 20.0
    max_response_bytes: int = 10 * 1024 * 1024
    allowed_mime_types: frozenset[str] = _ALLOWED_MIME_TYPES
    user_agent: str = "PRAgent/0.1 (safe web snapshot fetcher)"

    def __post_init__(self) -> None:
        if self.max_redirects < 0:
            raise ValueError("max_redirects 不能为负数")
        if self.timeout_seconds <= 0 or self.max_response_bytes <= 0:
            raise ValueError("timeout_seconds 和 max_response_bytes 必须大于 0")
        if not self.allowed_mime_types:
            raise ValueError("allowed_mime_types 不能为空")


@dataclass(frozen=True)
class ResolvedTarget:
    url: str
    scheme: str
    hostname: str
    port: int
    ip_address: str
    request_target: str
    host_header: str


@dataclass(frozen=True)
class TransportResponse:
    status: int
    headers: Mapping[str, str]
    body: bytes


@dataclass(frozen=True)
class SafeFetchResult:
    requested_url: str
    final_url: str
    status_code: int
    content_type: str
    body: bytes
    redirect_chain: tuple[str, ...]
    resolved_ips: tuple[str, ...]


class FetchTransport(Protocol):
    def request(
        self,
        target: ResolvedTarget,
        *,
        headers: Mapping[str, str],
        timeout: float,
        max_bytes: int,
    ) -> TransportResponse: ...


class SafeFetcher:
    def __init__(
        self,
        *,
        policy: Optional[FetchPolicy] = None,
        resolver: Optional[Callable[[str, int], Iterable[str]]] = None,
        transport: Optional[FetchTransport] = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.policy = policy or FetchPolicy()
        self.resolver = resolver or _resolve_addresses
        self.transport = transport or PinnedHTTPTransport()
        self._monotonic = monotonic

    def fetch(self, url: str) -> SafeFetchResult:
        requested = _strip_fragment(str(url).strip())
        if not requested:
            raise SafeFetchError("URL 不能为空", code="invalid_url")
        deadline = self._monotonic() + self.policy.timeout_seconds
        current = requested
        redirects: list[str] = []
        resolved_ips: list[str] = []
        visited: set[str] = set()

        for redirect_count in range(self.policy.max_redirects + 1):
            if current in visited:
                raise SafeFetchError("检测到重定向循环", code="redirect_loop")
            visited.add(current)
            remaining = deadline - self._monotonic()
            if remaining <= 0:
                raise SafeFetchError(
                    "网页抓取超过总超时限制",
                    code="timeout",
                    retryable=True,
                )
            target, addresses = _resolve_target(current, self.resolver)
            resolved_ips.extend(addresses)
            try:
                response = self.transport.request(
                    target,
                    headers={
                        "Accept": "text/html,application/xhtml+xml",
                        "Accept-Encoding": "identity",
                        "Connection": "close",
                        "Host": target.host_header,
                        "User-Agent": self.policy.user_agent,
                    },
                    timeout=remaining,
                    max_bytes=self.policy.max_response_bytes,
                )
            except SafeFetchError:
                raise
            except (OSError, http.client.HTTPException, ssl.SSLError) as exc:
                raise SafeFetchError(
                    f"网页请求失败：{exc}",
                    code="network_error",
                    retryable=True,
                ) from exc

            headers = {str(key).lower(): str(value) for key, value in response.headers.items()}
            if response.status in _REDIRECT_STATUSES:
                location = headers.get("location", "").strip()
                if not location:
                    raise SafeFetchError(
                        "重定向响应缺少 Location",
                        code="invalid_redirect",
                        status_code=response.status,
                    )
                if redirect_count >= self.policy.max_redirects:
                    raise SafeFetchError(
                        "网页重定向次数超过限制",
                        code="too_many_redirects",
                        status_code=response.status,
                    )
                next_url = _strip_fragment(urljoin(current, location))
                redirects.append(next_url)
                current = next_url
                continue

            if not 200 <= response.status < 300:
                raise SafeFetchError(
                    f"网页返回 HTTP {response.status}",
                    code=f"http_{response.status}",
                    retryable=response.status in {408, 425, 429, 500, 502, 503, 504},
                    status_code=response.status,
                )
            content_type = _content_type(headers.get("content-type"))
            if content_type not in self.policy.allowed_mime_types:
                raise SafeFetchError(
                    f"网页 MIME 不受支持：{content_type or 'missing'}",
                    code="invalid_mime",
                    status_code=response.status,
                )
            raw_length = headers.get("content-length")
            if raw_length is not None:
                try:
                    declared_length = int(raw_length)
                except ValueError as exc:
                    raise SafeFetchError(
                        "网页 Content-Length 无效",
                        code="invalid_content_length",
                    ) from exc
                if declared_length < 0:
                    raise SafeFetchError(
                        "网页 Content-Length 无效",
                        code="invalid_content_length",
                    )
                if declared_length > self.policy.max_response_bytes:
                    raise SafeFetchError(
                        "网页响应超过大小限制",
                        code="response_too_large",
                    )
            if len(response.body) > self.policy.max_response_bytes:
                raise SafeFetchError(
                    "网页响应超过大小限制",
                    code="response_too_large",
                )
            if not response.body:
                raise SafeFetchError("网页响应正文为空", code="empty_response")
            return SafeFetchResult(
                requested_url=requested,
                final_url=target.url,
                status_code=response.status,
                content_type=content_type,
                body=response.body,
                redirect_chain=tuple(redirects),
                resolved_ips=tuple(resolved_ips),
            )

        raise AssertionError("unreachable redirect loop")


class PinnedHTTPTransport:
    """Connect to the validated IP while retaining the logical Host and TLS SNI."""

    def __init__(self, *, ssl_context: Optional[ssl.SSLContext] = None) -> None:
        self.ssl_context = ssl_context or ssl.create_default_context()

    def request(
        self,
        target: ResolvedTarget,
        *,
        headers: Mapping[str, str],
        timeout: float,
        max_bytes: int,
    ) -> TransportResponse:
        if target.scheme == "https":
            connection: http.client.HTTPConnection = _PinnedHTTPSConnection(
                target.hostname,
                target.ip_address,
                target.port,
                timeout=timeout,
                context=self.ssl_context,
            )
        else:
            connection = http.client.HTTPConnection(
                target.ip_address,
                target.port,
                timeout=timeout,
            )
        try:
            connection.request("GET", target.request_target, headers=dict(headers))
            response = connection.getresponse()
            response_headers = {key.lower(): value for key, value in response.getheaders()}
            if response.status in _REDIRECT_STATUSES:
                body = b""
            else:
                raw_length = response_headers.get("content-length")
                if raw_length:
                    try:
                        length = int(raw_length)
                    except ValueError as exc:
                        raise SafeFetchError(
                            "网页 Content-Length 无效",
                            code="invalid_content_length",
                        ) from exc
                    if length < 0 or length > max_bytes:
                        raise SafeFetchError(
                            "网页响应超过大小限制",
                            code="response_too_large",
                        )
                body = response.read(max_bytes + 1)
                if len(body) > max_bytes:
                    raise SafeFetchError(
                        "网页响应超过大小限制",
                        code="response_too_large",
                    )
            return TransportResponse(response.status, response_headers, body)
        finally:
            connection.close()


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    def __init__(
        self,
        logical_host: str,
        pinned_ip: str,
        port: int,
        *,
        timeout: float,
        context: ssl.SSLContext,
    ) -> None:
        super().__init__(logical_host, port=port, timeout=timeout, context=context)
        self._pinned_ip = pinned_ip

    def connect(self) -> None:
        sock = socket.create_connection(
            (self._pinned_ip, self.port),
            self.timeout,
            self.source_address,
        )
        try:
            self.sock = self._context.wrap_socket(sock, server_hostname=self.host)
        except Exception:
            sock.close()
            raise


def _resolve_target(
    url: str,
    resolver: Callable[[str, int], Iterable[str]],
) -> tuple[ResolvedTarget, tuple[str, ...]]:
    if any(ord(character) < 32 or ord(character) == 127 for character in url):
        raise SafeFetchError("URL 包含控制字符", code="invalid_url")
    if "\\" in url:
        raise SafeFetchError("URL 包含反斜杠", code="invalid_url")
    try:
        parsed = urlsplit(url)
    except ValueError as exc:
        raise SafeFetchError("URL 格式无效", code="invalid_url") from exc
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"}:
        raise SafeFetchError("URL 只允许 http/https", code="invalid_scheme")
    if parsed.username is not None or parsed.password is not None:
        raise SafeFetchError("URL 不允许 credentials", code="url_credentials")
    hostname = (parsed.hostname or "").rstrip(".").lower()
    if not hostname:
        raise SafeFetchError("URL 缺少 host", code="invalid_url")
    if "%" in hostname:
        raise SafeFetchError("URL host 不允许 zone identifier", code="invalid_url")
    try:
        hostname = hostname.encode("idna").decode("ascii")
        port = parsed.port or (443 if scheme == "https" else 80)
    except (UnicodeError, ValueError) as exc:
        raise SafeFetchError("URL host/port 无效", code="invalid_url") from exc
    if not 1 <= port <= 65535:
        raise SafeFetchError("URL port 无效", code="invalid_url")
    try:
        raw_addresses = tuple(str(address) for address in resolver(hostname, port))
    except SafeFetchError:
        raise
    except Exception as exc:
        raise SafeFetchError(
            f"DNS 解析失败：{exc}", code="dns_error", retryable=True
        ) from exc
    if not raw_addresses:
        raise SafeFetchError("DNS 未返回地址", code="dns_error", retryable=True)
    addresses: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = []
    for raw_address in raw_addresses:
        try:
            address = ipaddress.ip_address(raw_address)
        except ValueError as exc:
            raise SafeFetchError("DNS 返回无效 IP", code="dns_error") from exc
        if not _is_public_address(address):
            raise SafeFetchError(
                "目标解析到非公网地址，已阻止请求",
                code="ssrf_blocked",
            )
        addresses.append(address)
    unique = tuple(
        str(address)
        for address in sorted(set(addresses), key=lambda item: (item.version, item.packed))
    )
    pinned_ip = unique[0]
    default_port = (scheme == "http" and port == 80) or (
        scheme == "https" and port == 443
    )
    host_display = f"[{hostname}]" if ":" in hostname else hostname
    host_header = host_display if default_port else f"{host_display}:{port}"
    encoded_path = quote(
        parsed.path or "/",
        safe="/:@-._~!$&'()*+,;=%",
    )
    encoded_query = quote(parsed.query, safe="=&/:?@-._~!$'()*+,;%")
    request_target = encoded_path
    if encoded_query:
        request_target += f"?{encoded_query}"
    normalized_url = urlunsplit((scheme, host_header, encoded_path, encoded_query, ""))
    return (
        ResolvedTarget(
            url=normalized_url,
            scheme=scheme,
            hostname=hostname,
            port=port,
            ip_address=pinned_ip,
            request_target=request_target,
            host_header=host_header,
        ),
        unique,
    )


def _resolve_addresses(hostname: str, port: int) -> tuple[str, ...]:
    rows = socket.getaddrinfo(
        hostname,
        port,
        family=socket.AF_UNSPEC,
        type=socket.SOCK_STREAM,
        proto=socket.IPPROTO_TCP,
    )
    return tuple(row[4][0] for row in rows)


def _is_public_address(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        return address.ipv4_mapped.is_global
    return address.is_global and not address.is_multicast and not address.is_unspecified


def _content_type(value: Optional[str]) -> str:
    return (value or "").split(";", 1)[0].strip().lower()


def _strip_fragment(url: str) -> str:
    try:
        parsed = urlsplit(url)
    except ValueError as exc:
        raise SafeFetchError("URL 格式无效", code="invalid_url") from exc
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, parsed.query, ""))
