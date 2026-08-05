from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import Mapping, Sequence
from typing import Any

from app.config.constants import Provider
from app.config.settings import settings
from app.gateway.providers.exceptions import ProviderError
from app.observability.logger import app_logger
from app.tools.base import BaseTool
from app.tools.executor import ToolExecutor
from app.tools.models import (
    ToolExecutionResult,
    ToolLoopResult,
    ToolPlan,
    ToolRound,
    ToolSchema,
)
from app.tools.prompts import (
    TOOL_PLANNER_SYSTEM_PROMPT,
    TOOL_SYNTHESIS_SYSTEM_PROMPT,
    build_planner_prompt,
    build_synthesis_prompt,
)
from app.tools.providers import (
    CalculatorTool,
    DateTimeTool,
    UUIDGeneratorTool,
    WeatherTool,
)
from app.tools.registry import ToolRegistry
from app.tools.types import LLMCaller, ToolLoopDecision

_JSON_FENCE_RE = re.compile(
    r"```(?:json)?\s*(.*?)\s*```",
    re.DOTALL | re.IGNORECASE,
)


def default_tools() -> list[BaseTool]:
    """Built-in tools registered when ToolService is constructed."""
    return [
        WeatherTool(),
        CalculatorTool(),
        DateTimeTool(),
        UUIDGeneratorTool(),
    ]


class ToolService:
    """
    High-level Tool Execution Framework API.

    Responsibilities
    - execute()
    - list_tools()
    - tool_schemas()
    - run() — multi-round tool loop (plan → execute → synthesize)

    Gateway communicates only with ToolService.
    New tool backends (REST, SQL, MCP, Slack, …) register into the
    injected ToolRegistry without Gateway changes.
    """

    def __init__(
        self,
        registry: ToolRegistry | None = None,
        executor: ToolExecutor | None = None,
        *,
        providers: Mapping[Provider, LLMCaller] | None = None,
        tools: Sequence[BaseTool] | None = None,
        enabled: bool | None = None,
        timeout_seconds: float | None = None,
        max_iterations: int | None = None,
        provider: Provider | None = None,
        model: str | None = None,
    ) -> None:
        self.registry = registry or ToolRegistry()
        self.executor = executor or ToolExecutor(
            self.registry,
            timeout_seconds=timeout_seconds,
        )
        self.providers = providers
        self.enabled = (
            settings.tools_enabled if enabled is None else enabled
        )
        self.max_iterations = (
            settings.max_tool_iterations
            if max_iterations is None
            else max_iterations
        )
        self.provider = provider or Provider(settings.tools_provider)
        self.model = model if model is not None else settings.tools_model

        seed = list(tools) if tools is not None else default_tools()
        for tool in seed:
            if tool.name not in self.registry:
                self.registry.register(tool)

    # ------------------------------------------------------------------
    # Public high-level API
    # ------------------------------------------------------------------

    async def list_tools(self) -> list[str]:
        return self.registry.list()

    async def tool_schemas(self) -> list[ToolSchema]:
        return self.registry.schemas()

    async def execute(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> ToolExecutionResult:
        return await self.executor.execute(name, arguments)

    async def execute_many(
        self,
        calls: Sequence[tuple[str, dict[str, Any]]],
        *,
        parallel: bool = True,
    ) -> list[ToolExecutionResult]:
        """
        Execute multiple tool calls.

        Independent calls run concurrently by default so multi-tool
        questions (date + calculate) stay fast.
        """
        if not calls:
            return []

        if parallel and len(calls) > 1:
            return list(
                await asyncio.gather(
                    *[
                        self.executor.execute(name, args)
                        for name, args in calls
                    ]
                )
            )

        results: list[ToolExecutionResult] = []
        for name, args in calls:
            results.append(await self.executor.execute(name, args))
        return results

    async def run(
        self,
        question: str,
        *,
        context_prompt: str | None = None,
        caller: LLMCaller | None = None,
        model: str | None = None,
    ) -> ToolLoopResult:
        """
        Full tool loop:

        1. Plan which tools to call (LLM)
        2. Execute tools (possibly many, in parallel)
        3. Synthesize final answer (LLM)
        4. Repeat if the model requests more tools

        If tools are disabled or planning says no tools are needed,
        falls back to a normal generation using context_prompt.
        """
        started = time.perf_counter()

        if not self.enabled or self.max_iterations <= 0:
            reason = (
                "tools_disabled"
                if not self.enabled
                else "max_tool_iterations_zero"
            )
            app_logger.info(f"Tool Loop Skipped reason={reason}")
            response = await self._plain_generate(
                question=question,
                context_prompt=context_prompt,
                caller=caller,
                model=model,
            )
            duration_ms = (time.perf_counter() - started) * 1000
            return ToolLoopResult(
                final_response=response,
                used_tools=False,
                duration_ms=duration_ms,
                skipped=True,
                skip_reason=reason,
            )

        schemas = await self.tool_schemas()
        if not schemas:
            app_logger.info("Tool Loop Skipped reason=no_tools_registered")
            response = await self._plain_generate(
                question=question,
                context_prompt=context_prompt,
                caller=caller,
                model=model,
            )
            duration_ms = (time.perf_counter() - started) * 1000
            return ToolLoopResult(
                final_response=response,
                used_tools=False,
                duration_ms=duration_ms,
                skipped=True,
                skip_reason="no_tools_registered",
            )

        llm = caller or self._resolve_caller()
        use_model = model if model is not None else self.model
        rounds: list[ToolRound] = []
        all_executions: list[ToolExecutionResult] = []
        used_tools = False

        app_logger.info(
            "Tool Loop Started "
            f"provider={(caller and 'injected') or self.provider.value} "
            f"model={use_model} "
            f"max_iterations={self.max_iterations} "
            f"tools={len(schemas)}"
        )

        try:
            plan = await self._plan(
                question=question,
                schemas=schemas,
                caller=llm,
                model=use_model,
            )

            for round_number in range(1, self.max_iterations + 1):
                if plan.decision != ToolLoopDecision.CALL_TOOLS:
                    if used_tools:
                        answer = (plan.answer or "").strip()
                        if not answer:
                            answer = self._fallback_answer_from_results(
                                question=question,
                                executions=all_executions,
                            )
                    else:
                        # No tools required — normal generation with
                        # full conversation / memory context.
                        answer = await self._plain_generate(
                            question=question,
                            context_prompt=context_prompt,
                            caller=llm,
                            model=use_model,
                        )
                    rounds.append(
                        ToolRound(
                            round_number=round_number,
                            plan=plan,
                            executions=[],
                        )
                    )
                    duration_ms = (time.perf_counter() - started) * 1000
                    app_logger.info(
                        f"Tool Loop Completed used_tools={used_tools} "
                        f"rounds={len(rounds)} "
                        f"duration_ms={duration_ms:.2f}"
                    )
                    return ToolLoopResult(
                        final_response=answer,
                        used_tools=used_tools,
                        rounds=rounds,
                        duration_ms=duration_ms,
                    )

                used_tools = True
                call_pairs = [
                    (item.tool, item.arguments) for item in plan.calls
                ]
                executions = await self.execute_many(call_pairs, parallel=True)
                all_executions.extend(executions)
                rounds.append(
                    ToolRound(
                        round_number=round_number,
                        plan=plan,
                        executions=executions,
                    )
                )

                app_logger.info(
                    f"Tool Round={round_number} "
                    f"calls={len(executions)} "
                    f"successes={sum(1 for e in executions if e.success)}"
                )

                if round_number >= self.max_iterations:
                    break

                plan = await self._synthesize(
                    question=question,
                    schemas=schemas,
                    executions=all_executions,
                    prior_context=context_prompt,
                    caller=llm,
                    model=use_model,
                )

            # Exhausted iterations while still requesting tools —
            # force a synthesis answer from whatever we have.
            final_plan = await self._synthesize(
                question=question,
                schemas=schemas,
                executions=all_executions,
                prior_context=context_prompt,
                caller=llm,
                model=use_model,
            )
            answer = (final_plan.answer or "").strip()
            if not answer:
                answer = self._fallback_answer_from_results(
                    question=question,
                    executions=all_executions,
                )

            duration_ms = (time.perf_counter() - started) * 1000
            app_logger.info(
                f"Tool Loop Completed used_tools={used_tools} "
                f"rounds={len(rounds)} "
                f"duration_ms={duration_ms:.2f}"
            )
            return ToolLoopResult(
                final_response=answer,
                used_tools=used_tools,
                rounds=rounds,
                duration_ms=duration_ms,
            )

        except Exception as exc:
            app_logger.error(
                f"Tool Loop failed; falling back to plain generation: {exc}"
            )
            response = await self._plain_generate(
                question=question,
                context_prompt=context_prompt,
                caller=caller or llm,
                model=model if model is not None else use_model,
            )
            duration_ms = (time.perf_counter() - started) * 1000
            return ToolLoopResult(
                final_response=response,
                used_tools=used_tools,
                rounds=rounds,
                duration_ms=duration_ms,
                skipped=True,
                skip_reason=f"tool_loop_error:{type(exc).__name__}",
            )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _plan(
        self,
        *,
        question: str,
        schemas: list[ToolSchema],
        caller: LLMCaller,
        model: str | None,
    ) -> ToolPlan:
        prompt = build_planner_prompt(question=question, schemas=schemas)
        raw = await caller.generate(
            prompt=prompt,
            system_prompt=TOOL_PLANNER_SYSTEM_PROMPT,
            model=model,
        )
        return self._parse_plan(raw)

    async def _synthesize(
        self,
        *,
        question: str,
        schemas: list[ToolSchema],
        executions: list[ToolExecutionResult],
        prior_context: str | None,
        caller: LLMCaller,
        model: str | None,
    ) -> ToolPlan:
        prompt = build_synthesis_prompt(
            question=question,
            schemas=schemas,
            executions=executions,
            prior_context=prior_context,
        )
        raw = await caller.generate(
            prompt=prompt,
            system_prompt=TOOL_SYNTHESIS_SYSTEM_PROMPT,
            model=model,
        )
        return self._parse_plan(raw)

    async def _plain_generate(
        self,
        *,
        question: str,
        context_prompt: str | None,
        caller: LLMCaller | None,
        model: str | None,
    ) -> str:
        llm = caller or self._resolve_caller()
        prompt = context_prompt or question
        return await llm.generate(prompt=prompt, model=model)

    def _resolve_caller(self) -> LLMCaller:
        if not self.providers:
            raise ProviderError(
                "ToolService has no LLM providers. "
                "Inject providers via DI or pass caller= to run()."
            )
        try:
            return self.providers[self.provider]
        except KeyError as exc:
            available = ", ".join(p.value for p in self.providers)
            raise ProviderError(
                f"Tools provider '{self.provider.value}' is not registered. "
                f"Available: {available}"
            ) from exc

    @staticmethod
    def _parse_plan(raw: str) -> ToolPlan:
        payload = ToolService._extract_json_object(raw)
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ProviderError(
                f"Tool planner returned invalid JSON: {raw[:200]!r}"
            ) from exc

        if not isinstance(data, dict):
            raise ProviderError("Tool planner JSON must be an object")

        try:
            return ToolPlan.model_validate(data)
        except Exception as exc:
            raise ProviderError(
                f"Tool planner JSON failed schema validation: {data!r}"
            ) from exc

    @staticmethod
    def _extract_json_object(raw: str) -> str:
        text = raw.strip()
        if not text:
            raise ProviderError("Tool planner returned empty response")

        fenced = _JSON_FENCE_RE.search(text)
        if fenced:
            text = fenced.group(1).strip()

        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise ProviderError(
                f"Tool planner response contained no JSON object: "
                f"{raw[:200]!r}"
            )
        return text[start : end + 1]

    @staticmethod
    def _fallback_answer_from_results(
        *,
        question: str,
        executions: list[ToolExecutionResult],
    ) -> str:
        if not executions:
            return (
                "I could not complete the tool requests needed to answer "
                f"your question: {question}"
            )

        lines = ["Here are the tool results I gathered:"]
        for item in executions:
            if item.success:
                lines.append(
                    f"- {item.tool}: {json.dumps(item.result, default=str)}"
                )
            else:
                lines.append(
                    f"- {item.tool}: failed ({item.error_code}: {item.error})"
                )
        return "\n".join(lines)
