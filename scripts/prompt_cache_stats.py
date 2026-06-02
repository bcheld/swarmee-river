#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


def _iter_jsonl_paths(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    if not path.exists():
        return
    for child in sorted(path.rglob("*.jsonl")):
        if child.is_file():
            yield child


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _extract_openai_cached_tokens(usage: dict[str, Any]) -> int | None:
    # OpenAI Responses API: usage.prompt_tokens_details.cached_tokens
    details = usage.get("prompt_tokens_details")
    if isinstance(details, dict):
        cached = _as_int(details.get("cached_tokens"))
        if cached is not None:
            return cached
    return None


def _extract_anthropic_cache_read_tokens(usage: dict[str, Any]) -> int | None:
    # Anthropic/Bedrock usage may use snake_case or camelCase.
    cached = _as_int(usage.get("cache_read_input_tokens"))
    if cached is None:
        cached = _as_int(usage.get("cacheReadInputTokens"))
    if cached is not None:
        return cached
    return None


def _extract_anthropic_cache_write_tokens(usage: dict[str, Any]) -> int | None:
    written = _as_int(usage.get("cache_write_input_tokens"))
    if written is None:
        written = _as_int(usage.get("cacheWriteInputTokens"))
    if written is None:
        written = _as_int(usage.get("cache_creation_input_tokens"))
    return written


def _extract_input_tokens(usage: dict[str, Any]) -> int | None:
    for key in ("input_tokens", "prompt_tokens", "inputTokens", "promptTokens"):
        v = _as_int(usage.get(key))
        if v is not None:
            return v
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize prompt caching usage from Swarmee JSONL logs.")
    parser.add_argument(
        "path",
        help="Path to a JSONL log file or a directory containing JSONL logs.",
    )
    args = parser.parse_args()

    root = Path(args.path).expanduser()
    total_calls = 0
    total_input_tokens = 0
    total_cached_tokens = 0
    total_cache_write_tokens = 0
    cache_events = 0

    for jsonl_path in _iter_jsonl_paths(root):
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if not isinstance(obj, dict):
                    continue
                if obj.get("event") != "after_model_call":
                    continue

                usage = obj.get("usage")
                if not isinstance(usage, dict):
                    continue

                total_calls += 1
                cached = _extract_openai_cached_tokens(usage) or _extract_anthropic_cache_read_tokens(usage) or 0
                cache_write = _extract_anthropic_cache_write_tokens(usage) or 0
                input_tokens = _extract_input_tokens(usage) or 0
                total_input_tokens += input_tokens + cached + cache_write

                if cached:
                    cache_events += 1
                total_cached_tokens += cached
                total_cache_write_tokens += cache_write

    hit_rate = (total_cached_tokens / total_input_tokens) if total_input_tokens else 0.0
    print("Prompt cache stats")
    print(f"- model calls: {total_calls}")
    print(f"- total input tokens: {total_input_tokens}")
    print(f"- cache read input tokens: {total_cached_tokens}")
    print(f"- cache write input tokens: {total_cache_write_tokens}")
    print(f"- cache usage events: {cache_events}")
    print(f"- cache read/total input ratio: {hit_rate:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
