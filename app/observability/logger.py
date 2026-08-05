from __future__ import annotations

import json
import sys
from contextvars import ContextVar
from typing import Any

from loguru import logger

from app.config.settings import settings
from app.observability.models import ObservabilityEvent, RequestSummary

# Bound to every log line for a single AI request.
request_id_ctx: ContextVar[str | None] = ContextVar("request_id", default=None)
session_id_ctx: ContextVar[str | None] = ContextVar("session_id", default=None)


def get_request_id() -> str | None:
    return request_id_ctx.get()


def bind_request_context(
    request_id: str,
    session_id: str | None = None,
) -> None:
    request_id_ctx.set(request_id)
    session_id_ctx.set(session_id)


def clear_request_context() -> None:
    request_id_ctx.set(None)
    session_id_ctx.set(None)


def _json_sink(message: Any) -> None:
    record = message.record
    payload: dict[str, Any] = {
        "timestamp": record["time"].isoformat(),
        "level": record["level"].name,
        "message": record["message"],
        "module": record["module"],
        "function": record["function"],
        "line": record["line"],
    }

    extra = record["extra"]
    if extra:
        payload.update(extra)

    rid = request_id_ctx.get()
    if rid and "request_id" not in payload:
        payload["request_id"] = rid

    sid = session_id_ctx.get()
    if sid and "session_id" not in payload:
        payload["session_id"] = sid

    sys.stdout.write(json.dumps(payload, default=str) + "\n")
    sys.stdout.flush()


def configure_logging() -> None:
    """Configure loguru for plain or JSON structured output."""
    logger.remove()

    if settings.enable_json_logging:
        logger.add(
            _json_sink,
            level=settings.log_level,
            format="{message}",
        )
    else:
        logger.add(
            sys.stdout,
            level=settings.log_level,
            format=(
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                "<level>{level: <8}</level> | "
                "{extra[request_id]} | "
                "{message}"
            ),
            filter=lambda record: record["extra"].setdefault(
                "request_id",
                request_id_ctx.get() or "-",
            )
            or True,
        )


configure_logging()

app_logger = logger


class ObservabilityLogger:
    """
    Structured logging facade used by ObservabilityService.

    Components should prefer ObservabilityService for request
    lifecycle signals; this logger emits the JSON lines.
    """

    def __init__(self, enabled: bool | None = None) -> None:
        self.enabled = (
            settings.observability_enabled if enabled is None else enabled
        )

    def info(self, message: str, **fields: Any) -> None:
        if not self.enabled:
            return
        app_logger.bind(**self._enrich(fields)).info(message)

    def warning(self, message: str, **fields: Any) -> None:
        if not self.enabled:
            return
        app_logger.bind(**self._enrich(fields)).warning(message)

    def error(self, message: str, **fields: Any) -> None:
        if not self.enabled:
            return
        app_logger.bind(**self._enrich(fields)).error(message)

    def log_event(self, event: ObservabilityEvent) -> None:
        if not self.enabled:
            return
        payload = event.model_dump(mode="json")
        app_logger.bind(**payload).info(
            event.message
            or f"event component={event.component.value} "
            f"type={event.event_type.value}"
        )

    def log_summary(self, summary: RequestSummary) -> None:
        if not self.enabled:
            return
        payload = summary.model_dump(mode="json", exclude={"events"})
        app_logger.bind(event_type="summary", **payload).info(
            f"request_summary request_id={summary.request_id} "
            f"status={summary.status.value} "
            f"latency_ms={summary.latency_ms:.2f} "
            f"provider={summary.provider} "
            f"route={summary.route} "
            f"tokens={summary.total_tokens} "
            f"cost_usd={summary.estimated_request_cost_usd}"
        )

    @staticmethod
    def _enrich(fields: dict[str, Any]) -> dict[str, Any]:
        enriched = dict(fields)
        rid = request_id_ctx.get()
        if rid and "request_id" not in enriched:
            enriched["request_id"] = rid
        sid = session_id_ctx.get()
        if sid and "session_id" not in enriched:
            enriched["session_id"] = sid
        return enriched
