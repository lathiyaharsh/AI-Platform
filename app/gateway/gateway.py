from app.config.constants import Provider
from app.gateway.providers.gemini_provider import GeminiProvider
from app.gateway.providers.groq_provider import GroqProvider
from app.gateway.router import Router
from app.prompts.manager import PromptManager
from asyncio import sleep

from app.config.settings import settings
from app.gateway.providers.exceptions import ProviderError

class Gateway:

    def __init__(self):
        self.router = Router()
        self.prompt_manager = PromptManager()
        self.providers = {
            Provider.GROQ: GroqProvider(),
            Provider.GEMINI: GeminiProvider(),
        }

    async def generate(
        self,
        prompt: str,
    ) -> str:

        decision = self.router.route(prompt)

        final_prompt = self.prompt_manager.render(
            decision.route.value,
            input=prompt,
        )

        provider = self.providers[decision.provider]

        for attempt in range(settings.max_retries):
            try:
                return await provider.generate(
                    prompt=final_prompt,
                    system_prompt=None,
                )
            except ProviderError:
                if attempt == settings.max_retries - 1:
                    break
                await sleep(settings.retry_delay)

        fallback = self.providers[Provider.GEMINI]
        return await fallback.generate(
            prompt=final_prompt,
        )
