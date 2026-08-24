from collections import deque

import pytest

import pragent.ingestion.safe_fetch as safe_fetch_module
from pragent.ingestion.safe_fetch import (
    FetchPolicy,
    SafeFetchError,
    SafeFetcher,
    TransportResponse,
)

PUBLIC_IP = "93.184.216.34"


class FakeTransport:
    def __init__(self, responses):
        self.responses = deque(responses)
        self.calls = []

    def request(self, target, *, headers, timeout, max_bytes):
        self.calls.append((target, dict(headers), timeout, max_bytes))
        response = self.responses.popleft()
        if isinstance(response, BaseException):
            raise response
        return response


def resolver_for(mapping):
    def resolve(host, port):
        value = mapping[host]
        return value() if callable(value) else value

    return resolve


def html_response(body=b"<html><article>public report text</article></html>", **headers):
    return TransportResponse(
        200,
        {"Content-Type": "text/html; charset=utf-8", **headers},
        body,
    )


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/admin",
        "http://10.0.0.1/",
        "http://169.254.169.254/latest/meta-data/",
        "http://[::1]/",
        "http://[fc00::1]/",
        "http://0.0.0.0/",
    ],
)
def test_private_special_and_metadata_addresses_are_blocked(url):
    host = url.split("//", 1)[1].split("/", 1)[0].strip("[]")
    fetcher = SafeFetcher(resolver=lambda hostname, port: [host])

    with pytest.raises(SafeFetchError) as captured:
        fetcher.fetch(url)
    assert captured.value.code == "ssrf_blocked"


def test_mixed_public_private_dns_answer_is_fail_closed_without_transport_call():
    transport = FakeTransport([html_response()])
    fetcher = SafeFetcher(
        resolver=lambda host, port: [PUBLIC_IP, "10.0.0.5"],
        transport=transport,
    )

    with pytest.raises(SafeFetchError) as captured:
        fetcher.fetch("https://example.org/report")
    assert captured.value.code == "ssrf_blocked"
    assert transport.calls == []


def test_redirect_is_revalidated_and_dns_rebinding_to_private_is_blocked():
    answers = iter([[PUBLIC_IP], ["127.0.0.1"]])
    transport = FakeTransport(
        [TransportResponse(302, {"Location": "/second"}, b"")]
    )
    fetcher = SafeFetcher(
        resolver=lambda host, port: next(answers),
        transport=transport,
    )

    with pytest.raises(SafeFetchError) as captured:
        fetcher.fetch("https://example.org/first")
    assert captured.value.code == "ssrf_blocked"
    assert len(transport.calls) == 1
    assert transport.calls[0][0].ip_address == PUBLIC_IP
    assert transport.calls[0][1]["Host"] == "example.org"


def test_safe_redirect_pins_each_validated_ip_and_returns_bounded_html():
    transport = FakeTransport(
        [
            TransportResponse(301, {"location": "https://cdn.example.org/article#x"}, b""),
            html_response(b"<html><article>final evidence text</article></html>"),
        ]
    )
    fetcher = SafeFetcher(
        resolver=resolver_for(
            {"example.org": [PUBLIC_IP], "cdn.example.org": ["1.1.1.1"]}
        ),
        transport=transport,
    )

    result = fetcher.fetch("https://example.org/start#fragment")

    assert result.requested_url == "https://example.org/start"
    assert result.final_url == "https://cdn.example.org/article"
    assert result.redirect_chain == ("https://cdn.example.org/article",)
    assert result.resolved_ips == (PUBLIC_IP, "1.1.1.1")
    assert transport.calls[0][0].ip_address == PUBLIC_IP
    assert transport.calls[1][0].ip_address == "1.1.1.1"
    assert result.content_type == "text/html"


def test_unicode_url_is_idna_and_percent_encoded_before_transport():
    transport = FakeTransport([html_response()])
    fetcher = SafeFetcher(
        resolver=lambda host, port: [PUBLIC_IP],
        transport=transport,
    )

    result = fetcher.fetch("https://例子.测试/研究报告?q=证据")

    target = transport.calls[0][0]
    assert target.hostname == "xn--fsqu00a.xn--0zwm56d"
    assert target.request_target == "/%E7%A0%94%E7%A9%B6%E6%8A%A5%E5%91%8A?q=%E8%AF%81%E6%8D%AE"
    assert result.final_url.endswith(target.request_target)


def test_credentials_scheme_redirect_limit_mime_size_and_status_are_rejected():
    fetcher = SafeFetcher(resolver=lambda host, port: [PUBLIC_IP])
    with pytest.raises(SafeFetchError) as captured:
        fetcher.fetch("file:///etc/passwd")
    assert captured.value.code == "invalid_scheme"
    with pytest.raises(SafeFetchError) as captured:
        fetcher.fetch("https://user:secret@example.org/")
    assert captured.value.code == "url_credentials"
    with pytest.raises(SafeFetchError) as captured:
        fetcher.fetch("https://[invalid/")
    assert captured.value.code == "invalid_url"

    wrong_mime = SafeFetcher(
        resolver=lambda host, port: [PUBLIC_IP],
        transport=FakeTransport(
            [TransportResponse(200, {"Content-Type": "application/pdf"}, b"%PDF")]
        ),
    )
    with pytest.raises(SafeFetchError) as captured:
        wrong_mime.fetch("https://example.org/")
    assert captured.value.code == "invalid_mime"

    too_large = SafeFetcher(
        policy=FetchPolicy(max_response_bytes=4),
        resolver=lambda host, port: [PUBLIC_IP],
        transport=FakeTransport([html_response(b"12345")]),
    )
    with pytest.raises(SafeFetchError) as captured:
        too_large.fetch("https://example.org/")
    assert captured.value.code == "response_too_large"

    declared_large = SafeFetcher(
        policy=FetchPolicy(max_response_bytes=4),
        resolver=lambda host, port: [PUBLIC_IP],
        transport=FakeTransport([html_response(b"x", **{"Content-Length": "5"})]),
    )
    with pytest.raises(SafeFetchError) as captured:
        declared_large.fetch("https://example.org/")
    assert captured.value.code == "response_too_large"

    unavailable = SafeFetcher(
        resolver=lambda host, port: [PUBLIC_IP],
        transport=FakeTransport([TransportResponse(503, {}, b"")]),
    )
    with pytest.raises(SafeFetchError) as captured:
        unavailable.fetch("https://example.org/")
    assert captured.value.code == "http_503" and captured.value.retryable

    redirecting = SafeFetcher(
        policy=FetchPolicy(max_redirects=1),
        resolver=lambda host, port: [PUBLIC_IP],
        transport=FakeTransport(
            [
                TransportResponse(302, {"Location": "/two"}, b""),
                TransportResponse(302, {"Location": "/three"}, b""),
            ]
        ),
    )
    with pytest.raises(SafeFetchError) as captured:
        redirecting.fetch("https://example.org/one")
    assert captured.value.code == "too_many_redirects"


def test_https_connection_uses_pinned_ip_but_logical_hostname_for_tls(monkeypatch):
    calls = {}

    class FakeSocket:
        def close(self):
            calls["closed"] = True

    class FakeContext:
        verify_mode = safe_fetch_module.ssl.CERT_REQUIRED
        check_hostname = True

        def wrap_socket(self, sock, *, server_hostname):
            calls["sni"] = server_hostname
            return sock

    def create_connection(address, timeout, source_address):
        calls["address"] = address
        calls["timeout"] = timeout
        return FakeSocket()

    monkeypatch.setattr(safe_fetch_module.socket, "create_connection", create_connection)
    connection = safe_fetch_module._PinnedHTTPSConnection(
        "example.org",
        PUBLIC_IP,
        443,
        timeout=7,
        context=FakeContext(),
    )
    connection.connect()

    assert calls["address"] == (PUBLIC_IP, 443)
    assert calls["sni"] == "example.org"
    assert calls["timeout"] == 7


def test_empty_dns_invalid_dns_and_network_errors_have_stable_codes():
    with pytest.raises(SafeFetchError) as captured:
        SafeFetcher(resolver=lambda host, port: []).fetch("https://example.org")
    assert captured.value.code == "dns_error"

    with pytest.raises(SafeFetchError) as captured:
        SafeFetcher(resolver=lambda host, port: ["not-an-ip"]).fetch(
            "https://example.org"
        )
    assert captured.value.code == "dns_error"

    fetcher = SafeFetcher(
        resolver=lambda host, port: [PUBLIC_IP],
        transport=FakeTransport([OSError("connection reset")]),
    )
    with pytest.raises(SafeFetchError) as captured:
        fetcher.fetch("https://example.org")
    assert captured.value.code == "network_error" and captured.value.retryable
