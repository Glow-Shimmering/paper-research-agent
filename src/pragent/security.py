"""Small security helpers shared by CLI and Web entry points."""
from __future__ import annotations

import hashlib
import hmac
import ipaddress
import secrets
from urllib.parse import urlsplit


def is_loopback_host(host: str) -> bool:
    """Return True only for host names/addresses confined to this machine."""
    normalized = host.strip().lower().strip("[]")
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def api_key_matches(provided: str | None, expected: str) -> bool:
    """Constant-time comparison; an empty configured key never authenticates."""
    return bool(expected and provided and secrets.compare_digest(provided, expected))


def ui_auth_token(api_key: str) -> str:
    """派生不暴露 API key 的 UI cookie 比较值。"""

    if not api_key:
        return ""
    return hmac.new(
        api_key.encode("utf-8"),
        b"pragent-ui-auth-v1",
        hashlib.sha256,
    ).hexdigest()


def ui_auth_matches(provided: str | None, api_key: str) -> bool:
    expected = ui_auth_token(api_key)
    return bool(
        expected
        and provided
        and secrets.compare_digest(provided, expected)
    )


def origin_matches_request(
    origin: str,
    *,
    request_scheme: str,
    request_host: str,
    request_port: int | None,
) -> bool:
    """严格比较浏览器 Origin 与当前请求，阻止跨站表单/脚本调用。"""
    try:
        parsed = urlsplit(origin)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
        ):
            return False
        origin_port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError:
        return False
    effective_request_port = request_port or (443 if request_scheme == "https" else 80)
    return (
        parsed.scheme == request_scheme
        and parsed.hostname.rstrip(".").lower() == request_host.rstrip(".").lower()
        and origin_port == effective_request_port
    )
