from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from tools.http_request import http_request


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        payload = {"path": self.path, "method": "GET"}
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length") or "0")
        data = self.rfile.read(length) if length else b""
        payload = {"path": self.path, "method": "POST", "body": data.decode("utf-8", errors="replace")}
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args, **_kwargs):  # pragma: no cover
        return


def _serve_in_thread() -> tuple[HTTPServer, int]:
    try:
        server = HTTPServer(("127.0.0.1", 0), _Handler)
    except PermissionError as exc:
        pytest.skip(f"Local HTTP server binding not permitted in this environment: {exc}")
    port = int(server.server_address[1])
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server, port


def _text(result: dict) -> str:
    return (result.get("content") or [{"text": ""}])[0].get("text", "")

def _response_body_json(result: dict) -> dict:
    text = _text(result)
    marker = "## Body"
    if marker not in text:
        raise AssertionError("Expected http_request output to include a '## Body' section")
    body = text.split(marker, 1)[1].lstrip()
    if body.startswith("\n"):
        body = body[1:]
    return json.loads(body.strip())


def test_http_request_get_with_params() -> None:
    server, port = _serve_in_thread()
    try:
        result = http_request(method="GET", url=f"http://127.0.0.1:{port}/hello", params={"q": "1"})
        assert result.get("status") == "success"
        payload = _response_body_json(result)
        assert payload["method"] == "GET"
        assert payload["path"].endswith("/hello?q=1")
    finally:
        server.shutdown()


def test_http_request_post_json_body() -> None:
    server, port = _serve_in_thread()
    try:
        result = http_request(method="POST", url=f"http://127.0.0.1:{port}/submit", json={"a": 1})
        assert result.get("status") == "success"
        payload = _response_body_json(result)
        assert payload["method"] == "POST"
        assert payload["path"].endswith("/submit")
        inner = json.loads(payload["body"])
        assert inner == {"a": 1}
    finally:
        server.shutdown()
