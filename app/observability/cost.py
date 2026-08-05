from __future__ import annotations

from threading import Lock
from typing import Mapping

from app.observability.models import CostEstimate, TokenUsage


# USD per 1M tokens. Update here (or inject custom table) without
# touching Gateway / business logic.
DEFAULT_PRICING: dict[str, dict[str, dict[str, float]]] = {
    "groq": {
        "llama-3.3-70b-versatile": {
            "input": 0.59,
            "output": 0.79,
        },
        "llama-3.1-8b-instant": {
            "input": 0.05,
            "output": 0.08,
        },
        "default": {
            "input": 0.59,
            "output": 0.79,
        },
    },
    "gemini": {
        "gemini-2.5-flash": {
            "input": 0.15,
            "output": 0.60,
        },
        "gemini-2.0-flash": {
            "input": 0.10,
            "output": 0.40,
        },
        "default": {
            "input": 0.15,
            "output": 0.60,
        },
    },
}

PRICING_VERSION = "v1"


class CostEstimator:
    """
    Estimates USD cost from token usage.

    Supports Groq and Gemini today. Future providers register
    pricing via ``update_pricing`` without Gateway changes.
    """

    def __init__(
        self,
        pricing: Mapping[str, Mapping[str, Mapping[str, float]]] | None = None,
        *,
        pricing_version: str = PRICING_VERSION,
    ) -> None:
        self._pricing: dict[str, dict[str, dict[str, float]]] = {
            provider: {model: dict(rates) for model, rates in models.items()}
            for provider, models in (pricing or DEFAULT_PRICING).items()
        }
        self.pricing_version = pricing_version
        self._session_costs: dict[str, float] = {}
        self._lock = Lock()

    def update_pricing(
        self,
        provider: str,
        model: str,
        *,
        input_per_million: float,
        output_per_million: float,
    ) -> None:
        """Hot-update rates for a provider/model pair."""
        key = provider.lower()
        with self._lock:
            self._pricing.setdefault(key, {})
            self._pricing[key][model] = {
                "input": input_per_million,
                "output": output_per_million,
            }

    def estimate(
        self,
        *,
        provider: str | None,
        model: str | None,
        tokens: TokenUsage,
        session_id: str | None = None,
    ) -> CostEstimate:
        rates = self._resolve_rates(provider, model)
        input_cost = (tokens.input_tokens / 1_000_000) * rates["input"]
        output_cost = (tokens.output_tokens / 1_000_000) * rates["output"]
        total = input_cost + output_cost

        if session_id:
            with self._lock:
                self._session_costs[session_id] = (
                    self._session_costs.get(session_id, 0.0) + total
                )

        return CostEstimate(
            provider=provider,
            model=model,
            input_cost_usd=round(input_cost, 8),
            output_cost_usd=round(output_cost, 8),
            total_cost_usd=round(total, 8),
            pricing_version=self.pricing_version,
        )

    def session_cost(self, session_id: str) -> float:
        with self._lock:
            return round(self._session_costs.get(session_id, 0.0), 8)

    def reset_session(self, session_id: str) -> None:
        with self._lock:
            self._session_costs.pop(session_id, None)

    def _resolve_rates(
        self,
        provider: str | None,
        model: str | None,
    ) -> dict[str, float]:
        if not provider:
            return {"input": 0.0, "output": 0.0}

        provider_key = provider.lower()
        models = self._pricing.get(provider_key)
        if not models:
            return {"input": 0.0, "output": 0.0}

        if model and model in models:
            return models[model]
        if "default" in models:
            return models["default"]

        # Fall back to first known model for the provider.
        first = next(iter(models.values()))
        return first
