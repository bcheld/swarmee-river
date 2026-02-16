from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Optional

from strands import tool


def _truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n… (truncated to {max_chars} chars) …"


def _normalize_headers(headers: dict[str, str] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    if not headers:
        return out
    for k, v in headers.items():
        key = str(k).strip()
        if not key:
            continue
        out[key] = str(v)
    return out


@tool
def http_request(
    *,
    method: str = "GET",
    url: str,
    params: Optional[dict[str, str]] = None,
    headers: Optional[dict[str, str]] = None,
    json: Optional[dict[str, Any]] = None,  # noqa: A002
    json_body: Optional[dict[str, Any]] = None,
    data: Optional[str] = None,
    timeout_s: int = 30,
    max_chars: int = 12000,
) -> dict[str, Any]:
    """
    Cross-platform fallback for `strands_tools.http_request`.

    Notes:
    - Uses Python stdlib `urllib.request`.
    - Supports query params, headers, and either JSON body or raw string body.
    """
    m = (method or "GET").strip().upper()
    if not url or not str(url).strip():
        return {"status": "error", "content": [{"text": "url is required"}]}

    try:
        parsed = urllib.parse.urlparse(url)
    except Exception:
        parsed = None

    if parsed is None or not parsed.scheme or not parsed.netloc:
        return {"status": "error", "content": [{"text": "url must be absolute (include scheme and host)"}]}

    final_url = url
    if params:
        query = urllib.parse.urlencode({str(k): str(v) for k, v in params.items()})
        if parsed.query:
            final_url = url + "&" + query
        else:
            final_url = url + "?" + query

    hdrs = _normalize_headers(headers)
    body_bytes: bytes | None = None
    effective_json = json_body if json_body is not None else json
    if (effective_json is not None) and data is not None:
        return {"status": "error", "content": [{"text": "Provide only one of json/json_body or data"}]}
    if json_body is not None and json is not None:
        return {"status": "error", "content": [{"text": "Provide only one of json or json_body"}]}
    if effective_json is not None:
        body_bytes = json.dumps(effective_json, ensure_ascii=False).encode("utf-8", errors="replace")
        hdrs.setdefault("Content-Type", "application/json; charset=utf-8")
    elif data is not None:
        body_bytes = str(data).encode("utf-8", errors="replace")
        hdrs.setdefault("Content-Type", "text/plain; charset=utf-8")

    # Simple allowlist: block file:// and other non-http schemes.
    if parsed.scheme.lower() not in {"http", "https"}:
        return {"status": "error", "content": [{"text": "Only http/https URLs are allowed"}]}

    req = urllib.request.Request(final_url, data=body_bytes, method=m, headers=hdrs)

    # Respect enterprise proxy config if present.
    proxy_handler = urllib.request.ProxyHandler()
    opener = urllib.request.build_opener(proxy_handler)

    try:
        with opener.open(req, timeout=max(1, int(timeout_s))) as resp:
            status = getattr(resp, "status", None)
            resp_headers = dict(resp.headers.items()) if getattr(resp, "headers", None) is not None else {}
            raw = resp.read() or b""
            # Best-effort decode.
            text = raw.decode("utf-8", errors="replace")
            summary = {
                "url": final_url,
                "status_code": int(status) if isinstance(status, int) else None,
                "bytes": len(raw),
            }
            content_text = "\n".join(
                [
                    "# HTTP Response",
                    f"- url: {summary['url']}",
                    f"- status_code: {summary['status_code']}",
                    f"- bytes: {summary['bytes']}",
                    "",
                    "## Headers",
                    json.dumps(resp_headers, indent=2, ensure_ascii=False, sort_keys=True),
                    "",
                    "## Body",
                    text,
                ]
            ).strip()
            return {"status": "success", "content": [{"text": _truncate(content_text, max_chars)}]}
    except urllib.error.HTTPError as e:
        try:
            raw = e.read() or b""
        except Exception:
            raw = b""
        text = raw.decode("utf-8", errors="replace") if raw else ""
        payload = f"HTTPError {getattr(e, 'code', None)}: {getattr(e, 'reason', '')}\n\n{text}".strip()
        return {"status": "error", "content": [{"text": _truncate(payload, max_chars)}]}
    except urllib.error.URLError as e:
        return {"status": "error", "content": [{"text": f"URLError: {e}"}]}
    except Exception as e:
        return {"status": "error", "content": [{"text": f"http_request failed: {e}"}]}
