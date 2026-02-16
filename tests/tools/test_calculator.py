from __future__ import annotations

from tools.calculator import calculator


def _text(result: dict) -> str:
    return (result.get("content") or [{"text": ""}])[0].get("text", "")


def test_calculator_basic_arithmetic() -> None:
    result = calculator("1 + 2 * 3")
    assert result.get("status") == "success"
    assert _text(result) == "7.0"


def test_calculator_math_functions() -> None:
    result = calculator("sqrt(9) + sin(0)")
    assert result.get("status") == "success"
    assert _text(result) == "3.0"


def test_calculator_rejects_names() -> None:
    result = calculator("__import__('os').system('echo hi')")
    assert result.get("status") == "error"
