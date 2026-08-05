from __future__ import annotations

from threading import Lock
from typing import Any

from app.observability.types import CacheKind


class PlatformMetrics:
    """
    Process-wide counters for cache / request health.

    In-memory today; a Prometheus exporter can scrape
    ``snapshot()`` later without Gateway changes.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._exact_hits = 0
        self._exact_misses = 0
        self._semantic_hits = 0
        self._semantic_misses = 0
        self._requests_total = 0
        self._requests_error = 0
        self._fallback_total = 0
        self._tool_calls_total = 0
        self._reflection_total = 0
        self._estimated_cost_usd = 0.0

    def record_cache(self, kind: CacheKind, *, hit: bool) -> None:
        with self._lock:
            if kind == CacheKind.EXACT:
                if hit:
                    self._exact_hits += 1
                else:
                    self._exact_misses += 1
            elif kind == CacheKind.SEMANTIC:
                if hit:
                    self._semantic_hits += 1
                else:
                    self._semantic_misses += 1

    def record_request(
        self,
        *,
        error: bool = False,
        fallback: bool = False,
        tool_calls: int = 0,
        reflection_used: bool = False,
        cost_usd: float = 0.0,
    ) -> None:
        with self._lock:
            self._requests_total += 1
            if error:
                self._requests_error += 1
            if fallback:
                self._fallback_total += 1
            self._tool_calls_total += tool_calls
            if reflection_used:
                self._reflection_total += 1
            self._estimated_cost_usd += cost_usd

    def hit_rate(self, kind: CacheKind) -> float:
        with self._lock:
            if kind == CacheKind.EXACT:
                total = self._exact_hits + self._exact_misses
                return self._exact_hits / total if total else 0.0
            total = self._semantic_hits + self._semantic_misses
            return self._semantic_hits / total if total else 0.0

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            exact_total = self._exact_hits + self._exact_misses
            semantic_total = self._semantic_hits + self._semantic_misses
            return {
                "requests_total": self._requests_total,
                "requests_error": self._requests_error,
                "fallback_total": self._fallback_total,
                "tool_calls_total": self._tool_calls_total,
                "reflection_total": self._reflection_total,
                "estimated_cost_usd": round(self._estimated_cost_usd, 8),
                "exact_cache_hits": self._exact_hits,
                "exact_cache_misses": self._exact_misses,
                "exact_cache_hit_rate": (
                    self._exact_hits / exact_total if exact_total else 0.0
                ),
                "semantic_cache_hits": self._semantic_hits,
                "semantic_cache_misses": self._semantic_misses,
                "semantic_cache_hit_rate": (
                    self._semantic_hits / semantic_total
                    if semantic_total
                    else 0.0
                ),
            }


def estimate_tokens(text: str) -> int:
    """
    Rough token estimate when a provider does not return usage.

    Uses ~4 characters per token — good enough for cost dashboards.
    """
    if not text:
        return 0
    return max(1, len(text) // 4)
