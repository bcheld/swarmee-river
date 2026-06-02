from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType


def _load_script() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / "prompt_cache_stats.py"
    spec = importlib.util.spec_from_file_location("prompt_cache_stats_script", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bedrock_camel_case_cache_usage_counts_total_input(tmp_path: Path, capsys, monkeypatch) -> None:
    script = _load_script()
    log_path = tmp_path / "events.jsonl"
    log_path.write_text(
        json.dumps(
            {
                "event": "after_model_call",
                "usage": {
                    "inputTokens": 100,
                    "outputTokens": 20,
                    "cacheReadInputTokens": 300,
                    "cacheWriteInputTokens": 50,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr("sys.argv", ["prompt_cache_stats.py", str(log_path)])

    assert script.main() == 0
    out = capsys.readouterr().out
    assert "- total input tokens: 450" in out
    assert "- cache read input tokens: 300" in out
    assert "- cache write input tokens: 50" in out
    assert "- cache read/total input ratio: 0.667" in out
