from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from app.observability.models import ObservabilityEvent, RequestSummary
from app.observability.types import Component


@runtime_checkable
class ObservabilityExporter(Protocol):
    """
    Pluggable sink for events and request summaries.

    Future implementations (without Gateway changes):
    - OpenTelemetryExporter
    - PrometheusExporter
    - LangfuseExporter
    - HeliconeExporter
    - PhoenixExporter
    - WandBExporter
    """

    name: str

    async def emit_event(self, event: ObservabilityEvent) -> None:
        ...

    async def emit_summary(self, summary: RequestSummary) -> None:
        ...


class LoggingExporter:
    """Default exporter — structured JSON via ObservabilityLogger."""

    name = "logging"

    def __init__(self, logger: Any) -> None:
        self._logger = logger

    async def emit_event(self, event: ObservabilityEvent) -> None:
        self._logger.log_event(event)

    async def emit_summary(self, summary: RequestSummary) -> None:
        self._logger.log_summary(summary)


class NullExporter:
    """No-op sink used when observability is disabled."""

    name = "null"

    async def emit_event(self, event: ObservabilityEvent) -> None:
        return None

    async def emit_summary(self, summary: RequestSummary) -> None:
        return None


class MetricsCollector:
    """
    Fan-out collector for observability events.

    Gateway talks only to ObservabilityService; the collector
    fans events out to registered exporters (logging today,
    OTel / Prometheus / Langfuse later).
    """

    def __init__(
        self,
        exporters: list[ObservabilityExporter] | None = None,
    ) -> None:
        self._exporters: list[ObservabilityExporter] = list(exporters or [])

    def register(self, exporter: ObservabilityExporter) -> None:
        self._exporters.append(exporter)

    def unregister(self, name: str) -> None:
        self._exporters = [e for e in self._exporters if e.name != name]

    @property
    def exporters(self) -> list[str]:
        return [e.name for e in self._exporters]

    async def collect(self, event: ObservabilityEvent) -> None:
        for exporter in self._exporters:
            await exporter.emit_event(event)

    async def collect_summary(self, summary: RequestSummary) -> None:
        for exporter in self._exporters:
            await exporter.emit_summary(summary)

    async def collect_error(
        self,
        *,
        request_id: str,
        component: Component,
        error: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        from app.observability.events import make_event
        from app.observability.types import EventType

        event = make_event(
            request_id=request_id,
            component=component,
            event_type=EventType.ERROR,
            metadata={"error": error, **(metadata or {})},
            message=error,
        )
        await self.collect(event)
