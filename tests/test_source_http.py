import json

import pytest

from pragent.sources.base import SourceProviderError
from pragent.sources.http import (
    HttpResponse,
    JsonHttpClient,
    RateLimiter,
    ResponseCache,
)


class FakeClock:
    def __init__(self):
        self.value = 1000.0
        self.sleeps = []

    def now(self):
        return self.value

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.value += seconds


def test_json_client_retries_429_obeys_retry_after_and_uses_disk_cache(tmp_path):
    clock = FakeClock()
    calls = []
    responses = [
        HttpResponse(429, {"Retry-After": "2"}, b'{"error":"slow down"}'),
        HttpResponse(200, {"Content-Type": "application/json"}, b'{"data":[1]}'),
    ]

    def requester(url, headers, timeout, max_bytes):
        calls.append((url, dict(headers), timeout, max_bytes))
        return responses.pop(0)

    cache = ResponseCache(tmp_path / "cache", wall_clock=clock.now)
    client = JsonHttpClient(
        "semantic_scholar",
        requester=requester,
        cache=cache,
        limiter=RateLimiter(1.0, monotonic=clock.now, sleep=clock.sleep),
        sleep=clock.sleep,
        wall_clock=clock.now,
    )
    url = "https://provider.example/search?q=rag"

    assert client.get_json(url, headers={"x-api-key": "secret"}) == {"data": [1]}
    assert len(calls) == 2
    assert clock.sleeps == [2.0]

    cached_client = JsonHttpClient(
        "semantic_scholar",
        requester=lambda *args: (_ for _ in ()).throw(AssertionError("network used")),
        cache=cache,
        limiter=RateLimiter(1.0, monotonic=clock.now, sleep=clock.sleep),
    )
    assert cached_client.get_json(url, headers={"x-api-key": "changed"}) == {
        "data": [1]
    }
    cache_text = "".join(path.read_text() for path in (tmp_path / "cache").rglob("*.json"))
    assert "secret" not in cache_text and "changed" not in cache_text


def test_json_client_bounded_failures_and_invalid_cache_are_fail_closed(tmp_path):
    cache = ResponseCache(tmp_path / "cache")
    bad_path = cache._path("crossref", "https://example.test")
    bad_path.parent.mkdir(parents=True)
    bad_path.write_text("not-json", encoding="utf-8")

    client = JsonHttpClient(
        "crossref",
        requester=lambda *args: HttpResponse(401, {}, b'{"message":"no"}'),
        cache=cache,
        max_retries=5,
    )
    with pytest.raises(SourceProviderError) as captured:
        client.get_json("https://example.test")
    assert captured.value.status_code == 401
    assert captured.value.retryable is False

    too_large = JsonHttpClient(
        "crossref",
        requester=lambda *args: HttpResponse(200, {}, b"{}x"),
        max_response_bytes=2,
    )
    with pytest.raises(SourceProviderError) as captured:
        too_large.get_json("https://example.test/large")
    assert captured.value.code == "response_too_large"

    invalid = JsonHttpClient(
        "crossref",
        requester=lambda *args: HttpResponse(200, {}, b"not json"),
    )
    with pytest.raises(SourceProviderError) as captured:
        invalid.get_json("https://example.test/bad")
    assert captured.value.code == "invalid_json"


def test_response_cache_ttl_and_envelope_are_deterministic(tmp_path):
    clock = FakeClock()
    cache = ResponseCache(tmp_path, ttl_seconds=5, wall_clock=clock.now)
    response = HttpResponse(200, {"ETag": "abc", "Authorization": "never"}, b"{}")
    cache.put("provider", "https://example.test", response)

    assert cache.get("provider", "https://example.test") == HttpResponse(
        200, {"etag": "abc"}, b"{}"
    )
    payload = json.loads(next(tmp_path.rglob("*.json")).read_text())
    assert payload["version"] == 1
    assert "authorization" not in payload["headers"]

    clock.value += 6
    assert cache.get("provider", "https://example.test") is None
