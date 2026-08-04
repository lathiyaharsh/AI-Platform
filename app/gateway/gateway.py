from app.config.constants import Provider
from app.gateway.providers.gemini_provider import GeminiProvider
from app.gateway.providers.groq_provider import GroqProvider
from app.gateway.router import Router

class Gateway:

    def __init__(self):
        self.router = Router()

        self.providers = {
            Provider.GROQ: GroqProvider(),
            Provider.GEMINI: GeminiProvider(),
        }

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> str:

        decision = self.router.route(prompt)

        provider = self.providers[decision.provider]

        return await provider.generate(
            prompt=prompt,
            system_prompt=system_prompt,
        )