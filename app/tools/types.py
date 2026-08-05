from enum import Enum
from typing import Any, Protocol, runtime_checkable


class ToolErrorCode(str, Enum):
    """
    Standardized tool failure codes.

    Future tools (REST, DB, MCP, Slack, …) map into these codes
    so Gateway never needs tool-specific error handling.
    """

    TOOL_NOT_FOUND = "tool_not_found"
    INVALID_ARGUMENTS = "invalid_arguments"
    TIMEOUT = "timeout"
    EXECUTION_ERROR = "execution_error"
    DISABLED = "disabled"
    UNKNOWN = "unknown"


class ToolLoopDecision(str, Enum):
    """Whether the planner wants tools or a final answer."""

    CALL_TOOLS = "call_tools"
    FINAL_ANSWER = "final_answer"


@runtime_checkable
class LLMCaller(Protocol):
    """
    Provider-agnostic generation interface.

    ToolService depends on this protocol so it never imports
    Groq/Gemini SDKs and can be tested with fakes.
    """

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model: str | None = None,
    ) -> str: ...


@runtime_checkable
class ExecutableTool(Protocol):
    """Minimal contract every registered tool must satisfy."""

    name: str
    description: str

    def schema(self) -> Any: ...

    async def execute(self, **kwargs: Any) -> Any: ...
