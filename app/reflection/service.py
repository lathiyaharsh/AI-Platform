from __future__ import annotations

import json
import re
import time
from collections.abc import Mapping

from app.config.constants import Provider
from app.config.settings import settings
from app.gateway.providers.exceptions import ProviderError
from app.observability.logger import app_logger
from app.reflection.models import (
    ReflectionPass,
    ReflectionResult,
    ReflectionVerdict,
)
from app.reflection.prompts import (
    IMPROVE_SYSTEM_PROMPT,
    REFLECTION_SYSTEM_PROMPT,
    build_improve_prompt,
    build_reflection_prompt,
)
from app.reflection.types import LLMCaller, ReflectionDecision

_JSON_FENCE_RE = re.compile(
    r"```(?:json)?\s*(.*?)\s*```",
    re.DOTALL | re.IGNORECASE,
)


class ReflectionService:
    """
    Evaluates model answers and optionally rewrites them.

    Independent of Gateway: inject providers via DI.
    Uses the shared provider abstraction (never provider SDKs).

    Designed so later phases can add:
    - multi-pass reflection
    - AI evaluation harnesses
    - human review queues
    - safety / guardrail checks
    """

    def __init__(
        self,
        providers: Mapping[Provider, LLMCaller],
        *,
        enabled: bool | None = None,
        provider: Provider | None = None,
        model: str | None = None,
        threshold: float | None = None,
        max_reflections: int | None = None,
    ) -> None:
        self.providers = providers
        self.enabled = (
            settings.reflection_enabled if enabled is None else enabled
        )
        self.provider = provider or Provider(settings.reflection_provider)
        self.model = model if model is not None else settings.reflection_model
        self.threshold = (
            settings.reflection_threshold if threshold is None else threshold
        )
        self.max_reflections = (
            settings.max_reflections
            if max_reflections is None
            else max_reflections
        )

    def should_improve(self, verdict: ReflectionVerdict) -> bool:
        """
        Decide whether the answer should be rewritten.

        Improve when the model requests it OR the quality score
        is below the configured threshold.
        """
        return verdict.improve or verdict.score < self.threshold

    async def reflect(
        self,
        question: str,
        answer: str,
    ) -> ReflectionResult:
        """
        Run up to max_reflections evaluate → (optional) improve cycles.
        """
        started = time.perf_counter()

        if not self.enabled or self.max_reflections <= 0:
            duration_ms = (time.perf_counter() - started) * 1000
            reason = (
                "reflection_disabled"
                if not self.enabled
                else "max_reflections_zero"
            )
            app_logger.info(f"Reflection Skipped reason={reason}")
            app_logger.info(f"Reflection Duration={duration_ms:.2f}ms")
            return ReflectionResult(
                original_response=answer,
                final_response=answer,
                improved=False,
                decision=ReflectionDecision.ACCEPT,
                duration_ms=duration_ms,
                skipped=True,
                skip_reason=reason,
            )

        app_logger.info(
            "Reflection Started "
            f"provider={self.provider.value} "
            f"model={self.model} "
            f"threshold={self.threshold} "
            f"max_reflections={self.max_reflections}"
        )

        current = answer
        passes: list[ReflectionPass] = []
        improved = False
        final_decision = ReflectionDecision.ACCEPT

        try:
            for pass_number in range(1, self.max_reflections + 1):
                verdict = await self._evaluate(
                    question=question,
                    answer=current,
                )

                app_logger.info(
                    f"Reflection Score={verdict.score:.4f} "
                    f"improve_flag={verdict.improve} "
                    f"pass={pass_number} "
                    f"reason={verdict.reason}"
                )

                if not self.should_improve(verdict):
                    app_logger.info(
                        "Reflection Skipped "
                        f"reason=quality_ok "
                        f"score={verdict.score:.4f} "
                        f"threshold={self.threshold}"
                    )
                    passes.append(
                        ReflectionPass(
                            pass_number=pass_number,
                            verdict=verdict,
                            decision=ReflectionDecision.ACCEPT,
                            response_before=current,
                            response_after=current,
                        )
                    )
                    final_decision = ReflectionDecision.ACCEPT
                    break

                before = current
                current = await self.improve(
                    question=question,
                    answer=before,
                )
                improved = True
                final_decision = ReflectionDecision.IMPROVE

                app_logger.info(
                    f"Reflection Improved pass={pass_number}"
                )

                passes.append(
                    ReflectionPass(
                        pass_number=pass_number,
                        verdict=verdict,
                        decision=ReflectionDecision.IMPROVE,
                        response_before=before,
                        response_after=current,
                    )
                )
        except Exception as exc:
            app_logger.error(f"Reflection failed; returning original: {exc}")
            duration_ms = (time.perf_counter() - started) * 1000
            app_logger.info(f"Reflection Duration={duration_ms:.2f}ms")
            return ReflectionResult(
                original_response=answer,
                final_response=answer,
                improved=False,
                decision=ReflectionDecision.ACCEPT,
                passes=passes,
                duration_ms=duration_ms,
                skipped=True,
                skip_reason=f"reflection_error:{type(exc).__name__}",
            )

        duration_ms = (time.perf_counter() - started) * 1000
        app_logger.info(f"Reflection Duration={duration_ms:.2f}ms")

        return ReflectionResult(
            original_response=answer,
            final_response=current,
            improved=improved,
            decision=final_decision,
            passes=passes,
            duration_ms=duration_ms,
            skipped=False,
        )

    async def improve(
        self,
        question: str,
        answer: str,
    ) -> str:
        """Rewrite an answer via the configured reflection provider."""
        caller = self._resolve_caller()
        prompt = build_improve_prompt(question=question, answer=answer)

        improved = await caller.generate(
            prompt=prompt,
            system_prompt=IMPROVE_SYSTEM_PROMPT,
            model=self.model,
        )

        cleaned = improved.strip()
        if not cleaned:
            raise ProviderError("Reflection improve returned empty response")
        return cleaned

    async def _evaluate(
        self,
        question: str,
        answer: str,
    ) -> ReflectionVerdict:
        caller = self._resolve_caller()
        prompt = build_reflection_prompt(question=question, answer=answer)

        raw = await caller.generate(
            prompt=prompt,
            system_prompt=REFLECTION_SYSTEM_PROMPT,
            model=self.model,
        )

        return self._parse_verdict(raw)

    def _resolve_caller(self) -> LLMCaller:
        try:
            return self.providers[self.provider]
        except KeyError as exc:
            available = ", ".join(p.value for p in self.providers)
            raise ProviderError(
                f"Reflection provider '{self.provider.value}' is not registered. "
                f"Available: {available}"
            ) from exc

    @staticmethod
    def _parse_verdict(raw: str) -> ReflectionVerdict:
        payload = ReflectionService._extract_json_object(raw)
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ProviderError(
                f"Reflection returned invalid JSON: {raw[:200]!r}"
            ) from exc

        if not isinstance(data, dict):
            raise ProviderError("Reflection JSON must be an object")

        try:
            return ReflectionVerdict.model_validate(data)
        except Exception as exc:
            raise ProviderError(
                f"Reflection JSON failed schema validation: {data!r}"
            ) from exc

    @staticmethod
    def _extract_json_object(raw: str) -> str:
        text = raw.strip()
        if not text:
            raise ProviderError("Reflection returned empty evaluation")

        fenced = _JSON_FENCE_RE.search(text)
        if fenced:
            text = fenced.group(1).strip()

        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise ProviderError(
                f"Reflection response contained no JSON object: {raw[:200]!r}"
            )
        return text[start : end + 1]
