from app.config.constants import Provider
from app.gateway.providers.gemini_provider import GeminiProvider
from app.gateway.providers.groq_provider import GroqProvider


class Gateway:
    """
    Central entry point for all LLM requests.
    """

    def __init__(self):
        self.providers = {
            Provider.GROQ: GroqProvider(),
            Provider.GEMINI: GeminiProvider(),
        }

    async def generate(
        self,
        prompt: str,
        provider: Provider = Provider.GROQ,
        system_prompt: str | None = None,
    ) -> str:

        selected_provider = self.providers[provider]

        return await selected_provider.generate(
            prompt=prompt,
            system_prompt=system_prompt,
        )