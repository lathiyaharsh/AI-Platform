from abc import ABC, abstractmethod


class BaseProvider(ABC):
    """Base class for all LLM providers."""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> str:
        """Generate a response from the LLM."""
        pass