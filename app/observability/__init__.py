"""
AI Observability Platform.

Gateway orchestrates request lifecycle through ObservabilityService.
Exporters (logging today; OTel / Prometheus / Langfuse later) plug in
via MetricsCollector without changing business logic.
"""

from app.observability.collector import (
    LoggingExporter,
    MetricsCollector,
    NullExporter,
    ObservabilityExporter,
)
from app.observability.cost import CostEstimator
from app.observability.logger import ObservabilityLogger, app_logger
from app.observability.metrics import PlatformMetrics, estimate_tokens
from app.observability.models import (
    ObservabilityEvent,
    RequestSummary,
    RequestTrace,
    TokenUsage,
    ToolCallMetric,
)
from app.observability.service import ObservabilityService
from app.observability.types import (
    CacheKind,
    Component,
    EventType,
    RequestStatus,
    TokenSource,
)

__all__ = [
    "CacheKind",
    "Component",
    "CostEstimator",
    "EventType",
    "LoggingExporter",
    "MetricsCollector",
    "NullExporter",
    "ObservabilityEvent",
    "ObservabilityExporter",
    "ObservabilityLogger",
    "ObservabilityService",
    "PlatformMetrics",
    "RequestStatus",
    "RequestSummary",
    "RequestTrace",
    "TokenSource",
    "TokenUsage",
    "ToolCallMetric",
    "app_logger",
    "estimate_tokens",
]
