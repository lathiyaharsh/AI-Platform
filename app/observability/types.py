from __future__ import annotations

from enum import Enum


class Component(str, Enum):
    """Platform component that emitted an observability event."""

    GATEWAY = "gateway"
    ROUTER = "router"
    MEMORY = "memory"
    LONG_TERM_MEMORY = "long_term_memory"
    RAG = "rag"
    CONTEXT_BUILDER = "context_builder"
    PROMPT_MANAGER = "prompt_manager"
    EXACT_CACHE = "exact_cache"
    SEMANTIC_CACHE = "semantic_cache"
    PROVIDER = "provider"
    TOOLS = "tools"
    REFLECTION = "reflection"
    EMBEDDINGS = "embeddings"
    COST = "cost"
    OBSERVABILITY = "observability"


class EventType(str, Enum):
    """Lifecycle / metric event kinds."""

    REQUEST_START = "request_start"
    REQUEST_END = "request_end"

    STAGE_START = "stage_start"
    STAGE_END = "stage_end"

    ROUTING = "routing"
    CACHE_HIT = "cache_hit"
    CACHE_MISS = "cache_miss"
    CACHE_SKIP = "cache_skip"
    CACHE_SET = "cache_set"

    RETRIEVAL = "retrieval"
    TOOL_CALL = "tool_call"
    TOOL_LOOP = "tool_loop"
    REFLECTION = "reflection"
    MEMORY = "memory"
    PROVIDER_CALL = "provider_call"
    RETRY = "retry"
    FALLBACK = "fallback"

    METRIC = "metric"
    ERROR = "error"
    SUMMARY = "summary"


class RequestStatus(str, Enum):
    """Terminal status for an AI request."""

    SUCCESS = "success"
    ERROR = "error"
    CACHE_HIT = "cache_hit"
    FALLBACK = "fallback"


class CacheKind(str, Enum):
    EXACT = "exact"
    SEMANTIC = "semantic"


class TokenSource(str, Enum):
    """Whether token counts came from the provider or were estimated."""

    PROVIDER = "provider"
    ESTIMATED = "estimated"
