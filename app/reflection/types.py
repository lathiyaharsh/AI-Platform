from enum import Enum
from typing import Protocol, runtime_checkable


class ReflectionDecision(str, Enum):
    """
    Outcome of a reflection pass.

    Future values (human review, safety reject, multi-pass) plug in here
    without changing Gateway orchestration.
    """

    ACCEPT = "accept"
    IMPROVE = "improve"
    HUMAN_REVIEW = "human_review"
    REJECT = "reject"


class ReflectionMode(str, Enum):
    """
    How aggressively reflection runs.

    EVALUATE: score only (no rewrite)
    AUTO_IMPROVE: evaluate + rewrite when below threshold
    MULTI_PASS / HUMAN_REVIEW reserved for later phases
    """

    DISABLED = "disabled"
    EVALUATE = "evaluate"
    AUTO_IMPROVE = "auto_improve"
    MULTI_PASS = "multi_pass"
    HUMAN_REVIEW = "human_review"


@runtime_checkable
class LLMCaller(Protocol):
    """
    Provider-agnostic generation interface.

    ReflectionService depends on this protocol so it never imports
    Groq/Gemini SDKs and can be tested with fakes.
    """

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model: str | None = None,
    ) -> str: ...
