from datetime import datetime, timezone

from pydantic import BaseModel, Field

from app.reflection.types import ReflectionDecision


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ReflectionVerdict(BaseModel):
    """
    Structured judgment from the reflection model.

    The reflection LLM must return JSON matching this schema.
    """

    improve: bool
    reason: str
    score: float = Field(ge=0.0, le=1.0)


class ReflectionPass(BaseModel):
    """Single evaluate (+ optional improve) cycle."""

    pass_number: int
    verdict: ReflectionVerdict
    decision: ReflectionDecision
    response_before: str
    response_after: str


class ReflectionResult(BaseModel):
    """
    Full reflection outcome returned to Gateway.

    Extensible for AI evaluation scores, guardrail flags,
    and human-review tickets in later phases.
    """

    original_response: str
    final_response: str
    improved: bool
    decision: ReflectionDecision
    passes: list[ReflectionPass] = Field(default_factory=list)
    duration_ms: float = 0.0
    skipped: bool = False
    skip_reason: str | None = None
    created_at: datetime = Field(default_factory=utc_now)

    @property
    def score(self) -> float | None:
        if not self.passes:
            return None
        return self.passes[-1].verdict.score
