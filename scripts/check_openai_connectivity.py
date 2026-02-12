from __future__ import annotations

import json
import os
import urllib.request
import urllib.error

from swarmee_river.utils.env_utils import load_env_file


def _normalize_openai_base_url(raw: str) -> str:
    base = (raw or "").strip().rstrip("/")
    if not base:
        return "https://api.openai.com/v1"
    # OpenAI Python SDK defaults to .../v1; accept both forms here.
    if base.endswith("/v1"):
        return base
    return base + "/v1"


def _extract_responses_text(data: dict) -> str:
    # Prefer the high-level convenience field if present
    text = data.get("output_text")
    if isinstance(text, str) and text.strip():
        return text.strip()

    # Fall back to output blocks
    output = data.get("output")
    if isinstance(output, list) and output:
        first = output[0]
        if isinstance(first, dict):
            content = first.get("content")
            if isinstance(content, list) and content:
                c0 = content[0]
                if isinstance(c0, dict):
                    t = c0.get("text")
                    if isinstance(t, str):
                        return t.strip()
    return ""


def _extract_chat_text(data: dict) -> str:
    choice0 = data.get("choices", [{}])[0]
    if isinstance(choice0, dict):
        msg = choice0.get("message", {})
        if isinstance(msg, dict):
            content = msg.get("content", "")
            if isinstance(content, str):
                return content.strip()
    return ""


def main() -> int:
    load_env_file()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Missing OPENAI_API_KEY (set it in .env or your environment).")
        return 2

    base_url = _normalize_openai_base_url(os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    model = os.getenv("SWARMEE_OPENAI_MODEL_ID", "gpt-5-nano")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Prefer the Responses API first (works for modern models).
    for endpoint, payload, extractor in [
        (
            "/responses",
            {
                "model": model,
                "input": "Reply with exactly: PONG",
                "max_output_tokens": 32,
            },
            _extract_responses_text,
        ),
        (
            "/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": "Reply with exactly: PONG"}],
                "max_completion_tokens": 32,
            },
            _extract_chat_text,
        ),
    ]:
        url = base_url + endpoint
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8", errors="replace")
            except Exception:
                body = ""
            print(f"{endpoint} failed: HTTP {e.code}")
            if body:
                print(body)
            continue
        except Exception as e:
            print(f"{endpoint} failed: {e}")
            continue

        text = extractor(data)
        print(text or json.dumps(data, indent=2)[:2000])
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
