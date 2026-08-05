"""
Prompts for tool planning and final-answer synthesis.

Kept as plain builders so PromptManager / versioning can adopt them later
without changing ToolService call sites.
"""

from __future__ import annotations

import json
from typing import Any

from app.tools.models import ToolExecutionResult, ToolSchema

TOOL_PLANNER_SYSTEM_PROMPT = (
    "You are a tool planner for a production AI platform. "
    "Decide whether the user question requires one or more tools. "
    "Respond with ONLY valid JSON. No markdown. No commentary."
)

TOOL_SYNTHESIS_SYSTEM_PROMPT = (
    "You are a helpful assistant for a production AI platform. "
    "Use the provided tool results to answer the user. "
    "If more tools are required, request them. "
    "Respond with ONLY valid JSON. No markdown. No commentary."
)


def build_planner_prompt(
    *,
    question: str,
    schemas: list[ToolSchema],
) -> str:
    schema_blob = json.dumps(
        [s.model_dump() for s in schemas],
        indent=2,
    )
    return (
        "Decide which tools (if any) are needed to answer the question.\n"
        "\n"
        "Available tools (JSON schemas):\n"
        f"{schema_blob}\n"
        "\n"
        "User question:\n"
        f"{question.strip()}\n"
        "\n"
        "Return ONLY JSON with this exact shape:\n"
        "{\n"
        '  "use_tools": true,\n'
        '  "calls": [\n'
        '    {"tool": "tool_name", "arguments": {}}\n'
        "  ],\n"
        '  "answer": null\n'
        "}\n"
        "\n"
        "Rules:\n"
        "- set use_tools=true and fill calls when tools are needed\n"
        "- set use_tools=false, calls=[], and put the direct answer in "
        '"answer" when no tools are needed\n'
        "- only use tool names from the available list\n"
        "- arguments must match each tool's parameters schema\n"
        "- you may request multiple tools in one response\n"
        "- do not invent tool results\n"
        "- do not include any text outside the JSON object\n"
    )


def build_synthesis_prompt(
    *,
    question: str,
    schemas: list[ToolSchema],
    executions: list[ToolExecutionResult],
    prior_context: str | None = None,
) -> str:
    schema_blob = json.dumps(
        [s.model_dump() for s in schemas],
        indent=2,
    )
    results_blob = json.dumps(
        [_execution_payload(item) for item in executions],
        indent=2,
        default=str,
    )
    context_block = ""
    if prior_context and prior_context.strip():
        context_block = (
            "Additional conversation context:\n"
            f"{prior_context.strip()}\n"
            "\n"
        )

    return (
        "Answer the user using the tool results below.\n"
        "If critical information is still missing, request more tools.\n"
        "\n"
        f"{context_block}"
        "Available tools (JSON schemas):\n"
        f"{schema_blob}\n"
        "\n"
        "User question:\n"
        f"{question.strip()}\n"
        "\n"
        "Tool results so far:\n"
        f"{results_blob}\n"
        "\n"
        "Return ONLY JSON with this exact shape:\n"
        "{\n"
        '  "use_tools": false,\n'
        '  "calls": [],\n'
        '  "answer": "final answer for the user"\n'
        "}\n"
        "\n"
        "Rules:\n"
        "- when you can answer, set use_tools=false and fill answer\n"
        "- when more tools are needed, set use_tools=true, fill calls, "
        "and set answer=null\n"
        "- base the answer on tool results; do not invent facts\n"
        "- be concise and helpful\n"
        "- do not include any text outside the JSON object\n"
    )


def _execution_payload(item: ToolExecutionResult) -> dict[str, Any]:
    return {
        "success": item.success,
        "tool": item.tool,
        "arguments": item.arguments,
        "result": item.result,
        "error": item.error,
        "error_code": item.error_code.value if item.error_code else None,
    }
