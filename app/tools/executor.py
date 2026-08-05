from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from pydantic import ValidationError

from app.config.settings import settings
from app.observability.logger import app_logger
from app.tools.models import ToolExecutionResult
from app.tools.registry import ToolRegistry
from app.tools.types import ToolErrorCode


class ToolExecutor:
    """
    Executes a single tool call with validation, timeout, and logging.

    Always returns a standardized ToolExecutionResult — never raises
    to callers. Gateway / ToolService only inspect success flags.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        *,
        timeout_seconds: float | None = None,
    ) -> None:
        self.registry = registry
        self.timeout_seconds = (
            settings.tool_timeout_seconds
            if timeout_seconds is None
            else timeout_seconds
        )

    async def execute(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> ToolExecutionResult:
        arguments = dict(arguments or {})
        started = time.perf_counter()

        app_logger.info(
            f"Tool Requested name={name} "
            f"arguments={_safe_repr(arguments)}"
        )

        tool = self.registry.get(name)
        if tool is None:
            duration_ms = _elapsed_ms(started)
            app_logger.error(
                f"Tool Executed Failed name={name} "
                f"error_code={ToolErrorCode.TOOL_NOT_FOUND.value} "
                f"duration_ms={duration_ms:.2f}"
            )
            return ToolExecutionResult(
                success=False,
                tool=name,
                error=f"Tool '{name}' not found",
                error_code=ToolErrorCode.TOOL_NOT_FOUND,
                duration_ms=duration_ms,
                arguments=arguments,
            )

        try:
            validated = tool.validate_arguments(arguments)
            payload = validated.model_dump()
        except ValidationError as exc:
            duration_ms = _elapsed_ms(started)
            app_logger.error(
                f"Tool Executed Failed name={name} "
                f"error_code={ToolErrorCode.INVALID_ARGUMENTS.value} "
                f"duration_ms={duration_ms:.2f} "
                f"detail={exc.error_count()} validation error(s)"
            )
            return ToolExecutionResult(
                success=False,
                tool=name,
                error=f"Invalid arguments: {exc}",
                error_code=ToolErrorCode.INVALID_ARGUMENTS,
                duration_ms=duration_ms,
                arguments=arguments,
            )

        try:
            result = await asyncio.wait_for(
                tool.execute(**payload),
                timeout=self.timeout_seconds,
            )
        except asyncio.TimeoutError:
            duration_ms = _elapsed_ms(started)
            app_logger.error(
                f"Tool Executed Failed name={name} "
                f"error_code={ToolErrorCode.TIMEOUT.value} "
                f"timeout={self.timeout_seconds}s "
                f"duration_ms={duration_ms:.2f}"
            )
            return ToolExecutionResult(
                success=False,
                tool=name,
                error=(
                    f"Tool '{name}' timed out after "
                    f"{self.timeout_seconds}s"
                ),
                error_code=ToolErrorCode.TIMEOUT,
                duration_ms=duration_ms,
                arguments=payload,
            )
        except Exception as exc:
            duration_ms = _elapsed_ms(started)
            app_logger.error(
                f"Tool Executed Failed name={name} "
                f"error_code={ToolErrorCode.EXECUTION_ERROR.value} "
                f"duration_ms={duration_ms:.2f} "
                f"error={type(exc).__name__}: {exc}"
            )
            return ToolExecutionResult(
                success=False,
                tool=name,
                error=f"{type(exc).__name__}: {exc}",
                error_code=ToolErrorCode.EXECUTION_ERROR,
                duration_ms=duration_ms,
                arguments=payload,
            )

        duration_ms = _elapsed_ms(started)
        result_size = _result_size(result)

        app_logger.info(
            f"Tool Executed name={name} "
            f"success=true "
            f"duration_ms={duration_ms:.2f} "
            f"result_size={result_size}"
        )

        return ToolExecutionResult(
            success=True,
            tool=name,
            result=result,
            duration_ms=duration_ms,
            arguments=payload,
            result_size=result_size,
        )


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000


def _result_size(result: Any) -> int:
    try:
        if isinstance(result, (str, bytes)):
            return len(result)
        return len(json.dumps(result, default=str))
    except Exception:
        return 0


def _safe_repr(value: Any, *, limit: int = 500) -> str:
    text = repr(value)
    if len(text) > limit:
        return text[:limit] + "…"
    return text
