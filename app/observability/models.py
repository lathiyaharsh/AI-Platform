from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Self

from pydantic import BaseModel, Field, model_validator

from app.observability.types import (
    Component,
    EventType,
    RequestStatus,
    TokenSource,
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ObservabilityEvent(BaseModel):
    """
    Standard event model for every AI-platform signal.

    Designed so OpenTelemetry / Langfuse / Helicone exporters can
    map fields without Gateway changes.
    """

    request_id: str
    timestamp: datetime = Field(default_factory=utc_now)
    component: Component
    event_type: EventType
    duration_ms: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    session_id: str | None = None
    user_id: str | None = None
    message: str | None = None


class TokenUsage(BaseModel):
    """Token accounting for a request (or a single LLM call)."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    source: TokenSource = TokenSource.ESTIMATED

    @model_validator(mode="after")
    def _ensure_total(self) -> Self:
        if self.total_tokens == 0:
            self.total_tokens = self.input_tokens + self.output_tokens
        return self


class CostEstimate(BaseModel):
    """USD cost estimate for a request / session."""

    provider: str | None = None
    model: str | None = None
    input_cost_usd: float = 0.0
    output_cost_usd: float = 0.0
    total_cost_usd: float = 0.0
    currency: str = "USD"
    pricing_version: str = "v1"


class ToolCallMetric(BaseModel):
    tool_name: str
    execution_time_ms: float = 0.0
    arguments: dict[str, Any] = Field(default_factory=dict)
    success: bool = True
    error: str | None = None


class RagMetrics(BaseModel):
    documents_retrieved: int = 0
    chunk_count: int = 0
    similarity_scores: list[float] = Field(default_factory=list)
    retrieval_time_ms: float = 0.0
    embedding_time_ms: float = 0.0
    vector_search_time_ms: float = 0.0
    context_size: int = 0
    success: bool = True
    error: str | None = None


class ReflectionMetrics(BaseModel):
    enabled: bool = False
    used: bool = False
    score: float | None = None
    improvement_performed: bool = False
    duration_ms: float = 0.0
    skipped: bool = False
    skip_reason: str | None = None
    pass_count: int = 0


class MemoryMetrics(BaseModel):
    session_id: str | None = None
    messages_loaded: int = 0
    messages_stored: int = 0
    sliding_window_size: int = 0


class RouterMetrics(BaseModel):
    chosen_route: str | None = None
    chosen_provider: str | None = None
    chosen_model: str | None = None
    routing_time_ms: float = 0.0


class CacheMetrics(BaseModel):
    exact_hit: bool = False
    exact_miss: bool = False
    semantic_hit: bool = False
    semantic_miss: bool = False
    semantic_skipped: bool = False


class RequestTrace(BaseModel):
    """
    Mutable in-flight request state collected by ObservabilityService.
    """

    request_id: str
    session_id: str | None = None
    user_id: str | None = None
    started_at: datetime = Field(default_factory=utc_now)
    ended_at: datetime | None = None
    status: RequestStatus = RequestStatus.SUCCESS

    provider: str | None = None
    model: str | None = None
    route: str | None = None
    prompt_version: int | None = None
    prompt_name: str | None = None

    retry_count: int = 0
    fallback_used: bool = False

    exact_cache_hit: bool = False
    semantic_cache_hit: bool = False
    reflection_used: bool = False

    tool_calls: list[ToolCallMetric] = Field(default_factory=list)
    tool_call_count: int = 0

    memory: MemoryMetrics = Field(default_factory=MemoryMetrics)
    rag: RagMetrics | None = None
    reflection: ReflectionMetrics | None = None
    router: RouterMetrics | None = None
    cache: CacheMetrics = Field(default_factory=CacheMetrics)

    tokens: TokenUsage = Field(default_factory=TokenUsage)
    cost: CostEstimate | None = None
    session_cost_usd: float = 0.0

    embedding_time_ms: float = 0.0
    vector_search_time_ms: float = 0.0
    reflection_time_ms: float = 0.0
    response_size: int = 0

    events: list[ObservabilityEvent] = Field(default_factory=list)
    extras: dict[str, Any] = Field(default_factory=dict)

    @property
    def latency_ms(self) -> float:
        if self.ended_at is None:
            return 0.0
        return (self.ended_at - self.started_at).total_seconds() * 1000


class RequestSummary(BaseModel):
    """
    Final observability summary for one AI request.

    Stable shape for Prometheus / Grafana / Langfuse / Phoenix exporters.
    """

    request_id: str
    session_id: str | None = None
    user_id: str | None = None

    request_start_time: datetime
    request_end_time: datetime
    latency_ms: float

    provider: str | None = None
    model: str | None = None
    route: str | None = None
    prompt_version: int | None = None
    prompt_name: str | None = None

    retry_count: int = 0
    fallback_used: bool = False

    exact_cache_hit: bool = False
    semantic_cache_hit: bool = False
    reflection_used: bool = False

    tool_calls: int = 0
    memory_messages: int = 0
    retrieved_chunks: int = 0

    embedding_time_ms: float = 0.0
    vector_search_time_ms: float = 0.0
    reflection_time_ms: float = 0.0
    response_size: int = 0
    status: RequestStatus = RequestStatus.SUCCESS

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    token_source: TokenSource = TokenSource.ESTIMATED

    estimated_request_cost_usd: float = 0.0
    estimated_session_cost_usd: float = 0.0

    events: list[ObservabilityEvent] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
