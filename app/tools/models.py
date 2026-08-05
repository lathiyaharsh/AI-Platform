from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from app.tools.types import ToolErrorCode, ToolLoopDecision


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ToolSchema(BaseModel):
    """
    JSON-schema description of a tool for LLM function calling.

    Shape is stable so later native provider tool APIs can reuse it.
    """

    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)


class ToolCallRequest(BaseModel):
    """A single planned tool invocation from the LLM planner."""

    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolPlan(BaseModel):
    """
    Structured planner output.

    The tool-decision LLM must return JSON matching this schema.
    """

    use_tools: bool = False
    calls: list[ToolCallRequest] = Field(default_factory=list)
    answer: str | None = None

    @property
    def decision(self) -> ToolLoopDecision:
        if self.use_tools and self.calls:
            return ToolLoopDecision.CALL_TOOLS
        return ToolLoopDecision.FINAL_ANSWER


class ToolExecutionResult(BaseModel):
    """
    Standardized executor response.

    Always returned — never raises to Gateway.
    """

    success: bool
    tool: str
    result: Any = None
    error: str | None = None
    error_code: ToolErrorCode | None = None
    duration_ms: float = 0.0
    arguments: dict[str, Any] = Field(default_factory=dict)
    result_size: int = 0
    created_at: datetime = Field(default_factory=utc_now)


class ToolRound(BaseModel):
    """One plan → execute cycle inside the tool loop."""

    round_number: int
    plan: ToolPlan
    executions: list[ToolExecutionResult] = Field(default_factory=list)


class ToolLoopResult(BaseModel):
    """
    Full tool-loop outcome returned to Gateway.

    Extensible for parallel batches, MCP sessions, and agent traces later.
    """

    final_response: str
    used_tools: bool = False
    rounds: list[ToolRound] = Field(default_factory=list)
    duration_ms: float = 0.0
    skipped: bool = False
    skip_reason: str | None = None
    created_at: datetime = Field(default_factory=utc_now)

    @property
    def execution_count(self) -> int:
        return sum(len(r.executions) for r in self.rounds)
