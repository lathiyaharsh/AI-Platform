from __future__ import annotations

import ast
import operator
from typing import Any, ClassVar

from pydantic import BaseModel, Field

from app.tools.base import BaseTool

_BINARY_OPS: dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}

_UNARY_OPS: dict[type, Any] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


class CalculatorParams(BaseModel):
    expression: str = Field(
        description=(
            "Arithmetic expression using +, -, *, /, and parentheses. "
            "Example: '(123 * 456) / 2'"
        ),
    )


class CalculatorTool(BaseTool):
    """
    Safe arithmetic evaluator.

    Uses AST walking — never eval() / exec().
    Supports +, -, *, /, parentheses, and unary +/-.
    """

    name: ClassVar[str] = "calculator"
    description: ClassVar[str] = (
        "Evaluate a safe arithmetic expression. "
        "Supports +, -, *, /, and parentheses. "
        "Does not support variables, functions, or exponentiation."
    )
    parameters_model: ClassVar[type[BaseModel]] = CalculatorParams

    async def execute(self, expression: str) -> dict[str, Any]:
        value = evaluate_expression(expression)
        return {
            "expression": expression.strip(),
            "result": value,
        }


def evaluate_expression(expression: str) -> float | int:
    text = expression.strip()
    if not text:
        raise ValueError("Expression cannot be empty")

    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid expression syntax: {expression!r}") from exc

    result = _eval_node(tree.body)

    if isinstance(result, float) and result.is_integer():
        return int(result)
    return result


def _eval_node(node: ast.AST) -> float | int:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(
            node.value, (int, float)
        ):
            raise ValueError("Only numeric literals are allowed")
        return node.value

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _UNARY_OPS:
            raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
        return _UNARY_OPS[op_type](_eval_node(node.operand))

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _BINARY_OPS:
            raise ValueError(
                f"Unsupported operator: {op_type.__name__}. "
                "Allowed: +, -, *, /"
            )
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if op_type is ast.Div and right == 0:
            raise ValueError("Division by zero")
        return _BINARY_OPS[op_type](left, right)

    if isinstance(node, ast.Expression):
        return _eval_node(node.body)

    raise ValueError(
        f"Unsupported expression node: {type(node).__name__}"
    )
