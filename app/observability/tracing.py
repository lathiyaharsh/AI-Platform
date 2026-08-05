from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from app.observability.models import ObservabilityEvent, RequestSummary


@runtime_checkable
class TraceExporter(Protocol):
    """
    Future distributed-tracing sink (OpenTelemetry spans).

    Kept separate from MetricsCollector so tracing backends
    (OTel, Langfuse, Phoenix, Helicone) can be registered later
    without changing Gateway or ObservabilityService call sites.
    """

    name: str

    async def on_event(self, event: ObservabilityEvent) -> None:
        ...

    async def on_summary(self, summary: RequestSummary) -> None:
        ...


class TracingBridge:
    """
    Optional bridge from ObservabilityService → TraceExporter list.

    No-op until an OpenTelemetry (or similar) exporter is registered.
    """

    def __init__(self, exporters: list[TraceExporter] | None = None) -> None:
        self._exporters: list[TraceExporter] = list(exporters or [])

    def register(self, exporter: TraceExporter) -> None:
        self._exporters.append(exporter)

    @property
    def enabled(self) -> bool:
        return bool(self._exporters)

    async def emit_event(self, event: ObservabilityEvent) -> None:
        for exporter in self._exporters:
            await exporter.on_event(event)

    async def emit_summary(self, summary: RequestSummary) -> None:
        for exporter in self._exporters:
            await exporter.on_summary(summary)

    def span_attributes(self, summary: RequestSummary) -> dict[str, Any]:
        """Canonical attribute map for future OTel spans."""
        return {
            "ai.request_id": summary.request_id,
            "ai.session_id": summary.session_id,
            "ai.provider": summary.provider,
            "ai.model": summary.model,
            "ai.route": summary.route,
            "ai.latency_ms": summary.latency_ms,
            "ai.input_tokens": summary.input_tokens,
            "ai.output_tokens": summary.output_tokens,
            "ai.total_tokens": summary.total_tokens,
            "ai.cost_usd": summary.estimated_request_cost_usd,
            "ai.exact_cache_hit": summary.exact_cache_hit,
            "ai.semantic_cache_hit": summary.semantic_cache_hit,
            "ai.fallback_used": summary.fallback_used,
            "ai.reflection_used": summary.reflection_used,
            "ai.tool_calls": summary.tool_calls,
            "ai.status": summary.status.value,
        }
