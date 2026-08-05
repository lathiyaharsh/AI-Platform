from __future__ import annotations

from typing import Any
from uuid import uuid4

from app.config.settings import settings
from app.observability.collector import (
    LoggingExporter,
    MetricsCollector,
    NullExporter,
    ObservabilityExporter,
)
from app.observability.cost import CostEstimator
from app.observability.events import make_event
from app.observability.logger import (
    ObservabilityLogger,
    bind_request_context,
    clear_request_context,
)
from app.observability.metrics import PlatformMetrics, estimate_tokens
from app.observability.models import (
    CacheMetrics,
    MemoryMetrics,
    ObservabilityEvent,
    RagMetrics,
    ReflectionMetrics,
    RequestSummary,
    RequestTrace,
    RouterMetrics,
    TokenUsage,
    ToolCallMetric,
    utc_now,
)
from app.observability.tracing import TracingBridge
from app.observability.types import (
    CacheKind,
    Component,
    EventType,
    RequestStatus,
    TokenSource,
)


class ObservabilityService:
    """
    Production AI observability facade.

    Responsibilities
    - start_request / end_request
    - record_event / record_metric
    - generate_summary
    - cost + token estimation
    - fan-out to MetricsCollector exporters

    Gateway is the only orchestrator that calls this service for
    the AI request lifecycle. Future sinks (OTel, Prometheus,
    Langfuse, Helicone, Phoenix, W&B) register as exporters —
    Gateway does not change.
    """

    def __init__(
        self,
        *,
        collector: MetricsCollector | None = None,
        cost_estimator: CostEstimator | None = None,
        metrics: PlatformMetrics | None = None,
        logger: ObservabilityLogger | None = None,
        tracing: TracingBridge | None = None,
        enabled: bool | None = None,
        estimate_cost: bool | None = None,
        log_request_body: bool | None = None,
        log_response_body: bool | None = None,
    ) -> None:
        self.enabled = (
            settings.observability_enabled if enabled is None else enabled
        )
        self.estimate_cost_enabled = (
            settings.estimate_cost if estimate_cost is None else estimate_cost
        )
        self.log_request_body = (
            settings.log_request_body
            if log_request_body is None
            else log_request_body
        )
        self.log_response_body = (
            settings.log_response_body
            if log_response_body is None
            else log_response_body
        )

        self.logger = logger or ObservabilityLogger(enabled=self.enabled)
        self.cost_estimator = cost_estimator or CostEstimator()
        self.metrics = metrics or PlatformMetrics()
        self.tracing = tracing or TracingBridge()

        if collector is not None:
            self.collector = collector
        elif self.enabled:
            self.collector = MetricsCollector(
                exporters=[LoggingExporter(self.logger)]
            )
        else:
            self.collector = MetricsCollector(exporters=[NullExporter()])

        self._active: dict[str, RequestTrace] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start_request(
        self,
        *,
        session_id: str | None = None,
        user_id: str | None = None,
        prompt: str | None = None,
        request_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Begin tracing an AI request. Returns request_id."""
        rid = request_id or str(uuid4())

        if not self.enabled:
            return rid

        bind_request_context(rid, session_id)

        meta = dict(metadata or {})
        if self.log_request_body and prompt is not None:
            meta["request_body"] = prompt
        elif prompt is not None:
            meta["request_chars"] = len(prompt)

        trace = RequestTrace(
            request_id=rid,
            session_id=session_id,
            user_id=user_id,
            memory=MemoryMetrics(session_id=session_id),
            extras=meta,
        )
        self._active[rid] = trace

        event = make_event(
            request_id=rid,
            component=Component.GATEWAY,
            event_type=EventType.REQUEST_START,
            session_id=session_id,
            user_id=user_id,
            metadata=meta,
            message="AI request started",
        )
        await self.record_event(event)
        return rid

    async def end_request(
        self,
        request_id: str,
        *,
        status: RequestStatus = RequestStatus.SUCCESS,
        response: str | None = None,
        input_text: str | None = None,
        output_text: str | None = None,
        tokens: TokenUsage | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> RequestSummary | None:
        """
        Finalize the request, estimate cost/tokens, emit summary.
        """
        if not self.enabled:
            clear_request_context()
            return None

        trace = self._active.get(request_id)
        if trace is None:
            clear_request_context()
            return None

        trace.ended_at = utc_now()
        trace.status = status

        if response is not None:
            trace.response_size = len(response)
            if self.log_response_body:
                trace.extras["response_body"] = response

        if tokens is not None:
            trace.tokens = tokens
        else:
            trace.tokens = self._resolve_tokens(
                input_text=input_text,
                output_text=output_text or response,
            )

        if self.estimate_cost_enabled:
            cost = self.cost_estimator.estimate(
                provider=trace.provider,
                model=trace.model,
                tokens=trace.tokens,
                session_id=trace.session_id,
            )
            trace.cost = cost
            if trace.session_id:
                trace.session_cost_usd = self.cost_estimator.session_cost(
                    trace.session_id
                )

        if metadata:
            trace.extras.update(metadata)

        self.metrics.record_request(
            error=status == RequestStatus.ERROR,
            fallback=trace.fallback_used,
            tool_calls=trace.tool_call_count,
            reflection_used=trace.reflection_used,
            cost_usd=(
                trace.cost.total_cost_usd if trace.cost is not None else 0.0
            ),
        )

        summary = self.generate_summary(trace)

        end_event = make_event(
            request_id=request_id,
            component=Component.GATEWAY,
            event_type=EventType.REQUEST_END,
            duration_ms=summary.latency_ms,
            session_id=trace.session_id,
            user_id=trace.user_id,
            metadata={
                "status": status.value,
                "latency_ms": summary.latency_ms,
                "total_tokens": summary.total_tokens,
                "cost_usd": summary.estimated_request_cost_usd,
            },
            message="AI request completed",
        )
        await self.record_event(end_event)
        await self.collector.collect_summary(summary)
        await self.tracing.emit_summary(summary)

        self._active.pop(request_id, None)
        clear_request_context()
        return summary

    def generate_summary(self, trace: RequestTrace) -> RequestSummary:
        """Build the canonical request summary from a completed trace."""
        ended = trace.ended_at or utc_now()
        cost = trace.cost
        return RequestSummary(
            request_id=trace.request_id,
            session_id=trace.session_id,
            user_id=trace.user_id,
            request_start_time=trace.started_at,
            request_end_time=ended,
            latency_ms=trace.latency_ms,
            provider=trace.provider,
            model=trace.model,
            route=trace.route,
            prompt_version=trace.prompt_version,
            prompt_name=trace.prompt_name,
            retry_count=trace.retry_count,
            fallback_used=trace.fallback_used,
            exact_cache_hit=trace.exact_cache_hit,
            semantic_cache_hit=trace.semantic_cache_hit,
            reflection_used=trace.reflection_used,
            tool_calls=trace.tool_call_count,
            memory_messages=trace.memory.messages_loaded,
            retrieved_chunks=(
                trace.rag.chunk_count if trace.rag is not None else 0
            ),
            embedding_time_ms=trace.embedding_time_ms,
            vector_search_time_ms=trace.vector_search_time_ms,
            reflection_time_ms=trace.reflection_time_ms,
            response_size=trace.response_size,
            status=trace.status,
            input_tokens=trace.tokens.input_tokens,
            output_tokens=trace.tokens.output_tokens,
            total_tokens=trace.tokens.total_tokens,
            token_source=trace.tokens.source,
            estimated_request_cost_usd=(
                cost.total_cost_usd if cost is not None else 0.0
            ),
            estimated_session_cost_usd=trace.session_cost_usd,
            events=list(trace.events),
            metadata={
                "prompt_name": trace.prompt_name,
                "cache": trace.cache.model_dump(),
                "router": (
                    trace.router.model_dump() if trace.router else None
                ),
                "rag": trace.rag.model_dump() if trace.rag else None,
                "reflection": (
                    trace.reflection.model_dump()
                    if trace.reflection
                    else None
                ),
                "memory": trace.memory.model_dump(),
                "tool_details": [
                    t.model_dump() for t in trace.tool_calls
                ],
                "platform_metrics": self.metrics.snapshot(),
                **trace.extras,
            },
        )

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    async def record_event(
        self,
        event: ObservabilityEvent | None = None,
        *,
        request_id: str | None = None,
        component: Component | None = None,
        event_type: EventType | None = None,
        duration_ms: float | None = None,
        metadata: dict[str, Any] | None = None,
        message: str | None = None,
    ) -> None:
        if not self.enabled:
            return

        if event is None:
            if request_id is None or component is None or event_type is None:
                raise ValueError(
                    "record_event requires an ObservabilityEvent or "
                    "request_id + component + event_type"
                )
            event = make_event(
                request_id=request_id,
                component=component,
                event_type=event_type,
                duration_ms=duration_ms,
                metadata=metadata,
                message=message,
            )

        trace = self._active.get(event.request_id)
        if trace is not None:
            if event.session_id is None:
                event.session_id = trace.session_id
            if event.user_id is None:
                event.user_id = trace.user_id
            trace.events.append(event)

        await self.collector.collect(event)
        await self.tracing.emit_event(event)

    async def record_metric(
        self,
        request_id: str,
        name: str,
        value: Any,
        *,
        component: Component = Component.OBSERVABILITY,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record a named metric onto the active request + event stream."""
        if not self.enabled:
            return

        trace = self._active.get(request_id)
        if trace is not None:
            trace.extras[name] = value

        await self.record_event(
            request_id=request_id,
            component=component,
            event_type=EventType.METRIC,
            metadata={"metric": name, "value": value, **(metadata or {})},
            message=f"metric {name}={value}",
        )

    # ------------------------------------------------------------------
    # Domain helpers (Gateway-friendly)
    # ------------------------------------------------------------------

    async def record_routing(
        self,
        request_id: str,
        *,
        route: str,
        provider: str,
        model: str,
        duration_ms: float,
    ) -> None:
        if not self.enabled:
            return

        trace = self._require_trace(request_id)
        trace.route = route
        trace.provider = provider
        trace.model = model
        trace.router = RouterMetrics(
            chosen_route=route,
            chosen_provider=provider,
            chosen_model=model,
            routing_time_ms=duration_ms,
        )
        await self.record_event(
            request_id=request_id,
            component=Component.ROUTER,
            event_type=EventType.ROUTING,
            duration_ms=duration_ms,
            metadata={
                "route": route,
                "provider": provider,
                "model": model,
            },
            message=(
                f"routed route={route} provider={provider} model={model}"
            ),
        )

    async def record_prompt(
        self,
        request_id: str,
        *,
        prompt_name: str,
        prompt_version: int,
    ) -> None:
        if not self.enabled:
            return

        trace = self._require_trace(request_id)
        trace.prompt_name = prompt_name
        trace.prompt_version = prompt_version
        await self.record_event(
            request_id=request_id,
            component=Component.PROMPT_MANAGER,
            event_type=EventType.METRIC,
            metadata={
                "prompt_name": prompt_name,
                "prompt_version": prompt_version,
            },
            message=(
                f"prompt name={prompt_name} version={prompt_version}"
            ),
        )

    async def record_memory(
        self,
        request_id: str,
        *,
        session_id: str,
        messages_loaded: int,
        messages_stored: int,
        sliding_window_size: int,
    ) -> None:
        if not self.enabled:
            return

        trace = self._require_trace(request_id)
        trace.memory = MemoryMetrics(
            session_id=session_id,
            messages_loaded=messages_loaded,
            messages_stored=messages_stored,
            sliding_window_size=sliding_window_size,
        )
        await self.record_event(
            request_id=request_id,
            component=Component.MEMORY,
            event_type=EventType.MEMORY,
            metadata=trace.memory.model_dump(),
            message=(
                f"memory loaded={messages_loaded} "
                f"stored={messages_stored} window={sliding_window_size}"
            ),
        )

    async def record_rag(
        self,
        request_id: str,
        *,
        documents_retrieved: int,
        chunk_count: int,
        similarity_scores: list[float],
        retrieval_time_ms: float,
        embedding_time_ms: float,
        context_size: int,
        success: bool = True,
        error: str | None = None,
    ) -> None:
        if not self.enabled:
            return

        vector_search_ms = max(0.0, retrieval_time_ms - embedding_time_ms)
        rag = RagMetrics(
            documents_retrieved=documents_retrieved,
            chunk_count=chunk_count,
            similarity_scores=similarity_scores,
            retrieval_time_ms=retrieval_time_ms,
            embedding_time_ms=embedding_time_ms,
            vector_search_time_ms=vector_search_ms,
            context_size=context_size,
            success=success,
            error=error,
        )
        trace = self._require_trace(request_id)
        trace.rag = rag
        trace.embedding_time_ms = embedding_time_ms
        trace.vector_search_time_ms = vector_search_ms

        await self.record_event(
            request_id=request_id,
            component=Component.RAG,
            event_type=EventType.RETRIEVAL,
            duration_ms=retrieval_time_ms,
            metadata=rag.model_dump(),
            message=(
                f"rag chunks={chunk_count} "
                f"context_size={context_size} "
                f"retrieval_ms={retrieval_time_ms:.2f}"
            ),
        )

    async def record_cache(
        self,
        request_id: str,
        *,
        kind: CacheKind,
        hit: bool,
        skipped: bool = False,
        duration_ms: float | None = None,
    ) -> None:
        if not self.enabled:
            return

        trace = self._require_trace(request_id)
        cache = trace.cache or CacheMetrics()

        if kind == CacheKind.EXACT:
            cache.exact_hit = hit
            cache.exact_miss = not hit and not skipped
            trace.exact_cache_hit = hit
        else:
            cache.semantic_hit = hit
            cache.semantic_miss = not hit and not skipped
            cache.semantic_skipped = skipped
            trace.semantic_cache_hit = hit

        trace.cache = cache

        if not skipped:
            self.metrics.record_cache(kind, hit=hit)

        if skipped:
            event_type = EventType.CACHE_SKIP
        elif hit:
            event_type = EventType.CACHE_HIT
        else:
            event_type = EventType.CACHE_MISS

        component = (
            Component.EXACT_CACHE
            if kind == CacheKind.EXACT
            else Component.SEMANTIC_CACHE
        )
        await self.record_event(
            request_id=request_id,
            component=component,
            event_type=event_type,
            duration_ms=duration_ms,
            metadata={
                "kind": kind.value,
                "hit": hit,
                "skipped": skipped,
                "exact_hit_rate": self.metrics.hit_rate(CacheKind.EXACT),
                "semantic_hit_rate": self.metrics.hit_rate(
                    CacheKind.SEMANTIC
                ),
            },
            message=(
                f"{kind.value} cache "
                f"{'SKIP' if skipped else ('HIT' if hit else 'MISS')}"
            ),
        )

    async def record_retry(
        self,
        request_id: str,
        *,
        attempt: int,
        error: str | None = None,
    ) -> None:
        if not self.enabled:
            return

        trace = self._require_trace(request_id)
        trace.retry_count = max(trace.retry_count, attempt)
        await self.record_event(
            request_id=request_id,
            component=Component.PROVIDER,
            event_type=EventType.RETRY,
            metadata={"attempt": attempt, "error": error},
            message=f"retry attempt={attempt}",
        )

    async def record_fallback(
        self,
        request_id: str,
        *,
        provider: str,
        model: str | None = None,
    ) -> None:
        if not self.enabled:
            return

        trace = self._require_trace(request_id)
        trace.fallback_used = True
        trace.provider = provider
        if model:
            trace.model = model
        await self.record_event(
            request_id=request_id,
            component=Component.PROVIDER,
            event_type=EventType.FALLBACK,
            metadata={"provider": provider, "model": model},
            message=f"fallback provider={provider}",
        )

    async def record_tools(
        self,
        request_id: str,
        *,
        used_tools: bool,
        duration_ms: float,
        executions: list[ToolCallMetric] | None = None,
        skipped: bool = False,
        skip_reason: str | None = None,
    ) -> None:
        if not self.enabled:
            return

        trace = self._require_trace(request_id)
        calls = executions or []
        trace.tool_calls = calls
        trace.tool_call_count = len(calls)

        await self.record_event(
            request_id=request_id,
            component=Component.TOOLS,
            event_type=EventType.TOOL_LOOP,
            duration_ms=duration_ms,
            metadata={
                "used_tools": used_tools,
                "tool_calls": len(calls),
                "skipped": skipped,
                "skip_reason": skip_reason,
                "executions": [c.model_dump() for c in calls],
            },
            message=(
                f"tools used={used_tools} calls={len(calls)} "
                f"duration_ms={duration_ms:.2f}"
            ),
        )

        for call in calls:
            await self.record_event(
                request_id=request_id,
                component=Component.TOOLS,
                event_type=EventType.TOOL_CALL,
                duration_ms=call.execution_time_ms,
                metadata=call.model_dump(),
                message=(
                    f"tool name={call.tool_name} "
                    f"success={call.success} "
                    f"duration_ms={call.execution_time_ms:.2f}"
                ),
            )

    async def record_reflection(
        self,
        request_id: str,
        *,
        enabled: bool,
        used: bool,
        score: float | None,
        improvement_performed: bool,
        duration_ms: float,
        skipped: bool = False,
        skip_reason: str | None = None,
        pass_count: int = 0,
    ) -> None:
        if not self.enabled:
            return

        reflection = ReflectionMetrics(
            enabled=enabled,
            used=used,
            score=score,
            improvement_performed=improvement_performed,
            duration_ms=duration_ms,
            skipped=skipped,
            skip_reason=skip_reason,
            pass_count=pass_count,
        )
        trace = self._require_trace(request_id)
        trace.reflection = reflection
        trace.reflection_used = used or improvement_performed
        trace.reflection_time_ms = duration_ms

        await self.record_event(
            request_id=request_id,
            component=Component.REFLECTION,
            event_type=EventType.REFLECTION,
            duration_ms=duration_ms,
            metadata=reflection.model_dump(),
            message=(
                f"reflection used={trace.reflection_used} "
                f"score={score} improved={improvement_performed} "
                f"duration_ms={duration_ms:.2f}"
            ),
        )

    def register_exporter(self, exporter: ObservabilityExporter) -> None:
        """Register OTel / Prometheus / Langfuse / … without Gateway edits."""
        self.collector.register(exporter)

    def platform_snapshot(self) -> dict[str, Any]:
        return self.metrics.snapshot()

    def get_trace(self, request_id: str) -> RequestTrace | None:
        return self._active.get(request_id)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _require_trace(self, request_id: str) -> RequestTrace:
        trace = self._active.get(request_id)
        if trace is None:
            # Soft-create so late events never crash the request path.
            trace = RequestTrace(request_id=request_id)
            self._active[request_id] = trace
        return trace

    def _resolve_tokens(
        self,
        *,
        input_text: str | None,
        output_text: str | None,
    ) -> TokenUsage:
        input_tokens = estimate_tokens(input_text or "")
        output_tokens = estimate_tokens(output_text or "")
        return TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            source=TokenSource.ESTIMATED,
        )
