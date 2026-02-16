from __future__ import annotations

import ast
import math
from typing import Any

from strands import tool

_ALLOWED_FUNCS: dict[str, Any] = {
    "abs": abs,
    "round": round,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "floor": math.floor,
    "ceil": math.ceil,
}

_ALLOWED_CONSTS: dict[str, Any] = {"pi": math.pi, "e": math.e, "tau": math.tau}


class _CalcError(ValueError):
    pass


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise _CalcError("Only numeric constants are allowed")

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _eval(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value

    if isinstance(node, ast.BinOp):
        left = _eval(node.left)
        right = _eval(node.right)
        op = node.op
        if isinstance(op, ast.Add):
            return left + right
        if isinstance(op, ast.Sub):
            return left - right
        if isinstance(op, ast.Mult):
            return left * right
        if isinstance(op, ast.Div):
            return left / right
        if isinstance(op, ast.FloorDiv):
            return left // right
        if isinstance(op, ast.Mod):
            return left % right
        if isinstance(op, ast.Pow):
            # Guardrail: avoid absurdly-large exponentiation.
            if abs(right) > 10_000:
                raise _CalcError("Exponent too large")
            return left**right
        raise _CalcError("Unsupported operator")

    if isinstance(node, ast.Name):
        name = str(node.id)
        if name in _ALLOWED_CONSTS:
            return float(_ALLOWED_CONSTS[name])
        raise _CalcError(f"Unknown identifier: {name}")

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise _CalcError("Only simple function calls are allowed")
        fn_name = str(node.func.id)
        fn = _ALLOWED_FUNCS.get(fn_name)
        if fn is None:
            raise _CalcError(f"Function not allowed: {fn_name}")
        if node.keywords:
            raise _CalcError("Keyword arguments are not allowed")
        args = [_eval(arg) for arg in node.args]
        return float(fn(*args))

    raise _CalcError("Unsupported expression")


@tool
def calculator(expression: str) -> dict[str, Any]:
    """
    Cross-platform fallback for `strands_tools.calculator`.

    Supports basic arithmetic and a small set of math functions/constants.
    """
    expr = (expression or "").strip()
    if not expr:
        return {"status": "error", "content": [{"text": "expression is required"}]}
    if len(expr) > 10_000:
        return {"status": "error", "content": [{"text": "expression too long"}]}

    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        return {"status": "error", "content": [{"text": f"Invalid expression: {e}"}]}

    try:
        value = _eval(tree)
    except _CalcError as e:
        return {"status": "error", "content": [{"text": str(e)}]}
    except Exception as e:
        return {"status": "error", "content": [{"text": f"calculation failed: {e}"}]}

    return {"status": "success", "content": [{"text": str(value)}]}
