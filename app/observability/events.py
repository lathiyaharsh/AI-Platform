from __future__ import annotations

from typing import Any

from app.observability.models import ObservabilityEvent
from app.observability.types import Component, EventType


def make_event(
    *,
    request_id: str,
    component: Component,
    event_type: EventType,
    duration_ms: float | None = None,
    metadata: dict[str, Any] | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
    message: str | None = None,
) -> ObservabilityEvent:
    """Factory for the standard observability event model."""
    return ObservabilityEvent(
        request_id=request_id,
        component=component,
        event_type=event_type,
        duration_ms=duration_ms,
        metadata=metadata or {},
        session_id=session_id,
        user_id=user_id,
        message=message,
    )


def stage_start(
    request_id: str,
    component: Component,
    *,
    session_id: str | None = None,
    user_id: str | None = None,
    **metadata: Any,
) -> ObservabilityEvent:
    return make_event(
        request_id=request_id,
        component=component,
        event_type=EventType.STAGE_START,
        session_id=session_id,
        user_id=user_id,
        metadata=metadata,
        message=f"{component.value} started",
    )


def stage_end(
    request_id: str,
    component: Component,
    *,
    duration_ms: float | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
    **metadata: Any,
) -> ObservabilityEvent:
    return make_event(
        request_id=request_id,
        component=component,
        event_type=EventType.STAGE_END,
        duration_ms=duration_ms,
        session_id=session_id,
        user_id=user_id,
        metadata=metadata,
        message=f"{component.value} completed",
    )
