from pragent.security import (
    api_key_matches,
    is_loopback_host,
    origin_matches_request,
    ui_auth_matches,
    ui_auth_token,
)


def test_loopback_detection_is_fail_closed():
    assert is_loopback_host("127.0.0.1")
    assert is_loopback_host("::1")
    assert is_loopback_host("[::1]")
    assert is_loopback_host("localhost")
    assert not is_loopback_host("0.0.0.0")
    assert not is_loopback_host("::")
    assert not is_loopback_host("pragent.local")


def test_api_key_comparison_requires_non_empty_expected_key():
    assert api_key_matches("secret", "secret")
    assert not api_key_matches("wrong", "secret")
    assert not api_key_matches(None, "secret")
    assert not api_key_matches("", "")


def test_ui_auth_cookie_is_derived_without_exposing_api_key():
    token = ui_auth_token("secret")

    assert token and token != "secret" and len(token) == 64
    assert ui_auth_matches(token, "secret")
    assert not ui_auth_matches(token, "changed")
    assert not ui_auth_matches(None, "secret")
    assert ui_auth_token("") == ""


def test_origin_must_match_scheme_host_and_port():
    kwargs = {
        "request_scheme": "http",
        "request_host": "127.0.0.1",
        "request_port": 8000,
    }
    assert origin_matches_request("http://127.0.0.1:8000", **kwargs)
    assert not origin_matches_request("https://127.0.0.1:8000", **kwargs)
    assert not origin_matches_request("http://127.0.0.1:9000", **kwargs)
    assert not origin_matches_request("https://evil.example", **kwargs)
    assert not origin_matches_request("null", **kwargs)
