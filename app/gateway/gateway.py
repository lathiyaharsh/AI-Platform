from app.config.constants import Provider
from app.gateway.providers.gemini_provider import GeminiProvider
from app.gateway.providers.groq_provider import GroqProvider
from app.gateway.router import Router
from app.prompts.manager import PromptManager
from asyncio import sleep

from app.config.settings import settings
from app.gateway.providers.exceptions import ProviderError
from app.cache.exact_cache import ExactCache

class Gateway:

    def __init__(self):
        self.router = Router()
        self.prompt_manager = PromptManager()
        self.cache = ExactCache()
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

        cache_key = f"gateway:{decision.provider}:{final_prompt}"
        cached = self.cache.get(cache_key)

        if cached:
            return cached

        for attempt in range(settings.max_retries):
            try:
                response = await provider.generate(
                    prompt=final_prompt,
                    system_prompt=None,
                )
                self.cache.set(cache_key, response)
                return response
            except ProviderError:
                if attempt == settings.max_retries - 1:
                    break
                await sleep(settings.retry_delay)

        fallback = self.providers[Provider.GEMINI]
        response = await fallback.generate(
            prompt=final_prompt,
        )
        self.cache.set(cache_key, response)
        return response