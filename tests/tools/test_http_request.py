from __future__ import annotations

import json
from typing import Any

import tools.http_request as http_request_module


def _text(result: dict) -> str:
    return (result.get("content") or [{"text": ""}])[0].get("text", "")


def _response_body_json(result: dict) -> dict:
    text = _text(result)
    marker = "## Body"
    if marker not in text:
        raise AssertionError("Expected http_request output to include a '## Body' section")
    body = text.split(marker, 1)[1].lstrip()
    return json.loads(body.strip())


class _FakeResponse:
    def __init__(self, *, status: int, headers: dict[str, str] | None = None, body: bytes = b"") -> None:
        self.status = status
        self.headers = headers or {}
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *_exc: object) -> None:
        return None


class _FakeOpener:
    def __init__(self, response: _FakeResponse) -> None:
        self.response = response
        self.last_request: Any | None = None
        self.last_timeout: int | None = None

    def open(self, req: Any, *, timeout: int) -> _FakeResponse:
        self.last_request = req
        self.last_timeout = timeout
        return self.response


def test_http_request_get_with_params(monkeypatch) -> None:
    response = _FakeResponse(
        status=200,
        headers={"Content-Type": "application/json"},
        body=json.dumps({"ok": True}).encode("utf-8"),
    )
    opener = _FakeOpener(response)

    monkeypatch.setattr(http_request_module.urllib.request, "build_opener", lambda *_args, **_kwargs: opener)

    result = http_request_module.http_request(method="GET", url="http://example.com/hello", params={"q": "1"})
    assert result.get("status") == "success"

    req = opener.last_request
    assert req is not None
    assert req.get_method() == "GET"
    assert req.full_url.endswith("/hello?q=1")
    assert getattr(req, "data", None) is None

    payload = _response_body_json(result)
    assert payload == {"ok": True}


def test_http_request_post_json_body(monkeypatch) -> None:
    response = _FakeResponse(
        status=200,
        headers={"Content-Type": "application/json"},
        body=json.dumps({"ok": True}).encode("utf-8"),
    )
    opener = _FakeOpener(response)

    monkeypatch.setattr(http_request_module.urllib.request, "build_opener", lambda *_args, **_kwargs: opener)

    result = http_request_module.http_request(method="POST", url="http://example.com/submit", json={"a": 1})
    assert result.get("status") == "success"

    req = opener.last_request
    assert req is not None
    assert req.get_method() == "POST"
    assert req.full_url.endswith("/submit")
    assert req.data is not None
    body = req.data.decode("utf-8", errors="replace")
    assert json.loads(body) == {"a": 1}

    # Content-Type should be set for JSON bodies.
    headers = {k.lower(): v for k, v in dict(getattr(req, "headers", {}) or {}).items()}
    assert "content-type" in headers
    assert "application/json" in headers["content-type"].lower()

    payload = _response_body_json(result)
    assert payload == {"ok": True}
