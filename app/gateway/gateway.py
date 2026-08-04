from asyncio import sleep

from app.cache.exact_cache import ExactCache
from app.cache.semantic_cache import SemanticCache
from app.config.constants import Provider
from app.config.settings import settings
from app.gateway.router import Router
from app.gateway.providers.exceptions import ProviderError
from app.gateway.providers.gemini_provider import GeminiProvider
from app.gateway.providers.groq_provider import GroqProvider
from app.observability.logger import app_logger
from app.prompts.manager import PromptManager


class Gateway:
    """
    Central AI Gateway.

    Responsibilities

    - Routing
    - Prompt rendering
    - Exact cache
    - Semantic cache
    - Retry
    - Fallback
    """

    def __init__(self):

        self.router = Router()

        self.prompt_manager = PromptManager()

        self.exact_cache = ExactCache()

        self.semantic_cache = SemanticCache()

        self.providers = {
            Provider.GROQ: GroqProvider(),
            Provider.GEMINI: GeminiProvider(),
        }

    async def generate(
        self,
        prompt: str,
    ) -> str:

        #
        # 1. Exact Cache
        #

        cached = self.exact_cache.get(prompt)

        if cached:

            app_logger.info("Exact Cache HIT")

            return cached

        #
        # 2. Semantic Cache
        #

        cached = await self.semantic_cache.get(prompt)

        if cached:

            app_logger.info("Semantic Cache HIT")

            self.exact_cache.set(prompt, cached)

            return cached

        #
        # 3. Intelligent Routing
        #

        decision = self.router.route(prompt)

        provider = self.providers[decision.provider]

        #
        # 4. Prompt Rendering
        #

        final_prompt = self.prompt_manager.render(
            decision.route.value,
            input=prompt,
        )

        #
        # 5. Retry
        #

        for attempt in range(settings.max_retries):

            try:

                app_logger.info(
                    f"Provider={decision.provider.value} "
                    f"Attempt={attempt+1}"
                )

                response = await provider.generate(
                    prompt=final_prompt,
                )

                #
                # Save to caches
                #

                self.exact_cache.set(
                    prompt,
                    response,
                )

                await self.semantic_cache.set(
                    prompt,
                    response,
                )

                return response

            except ProviderError as e:

                app_logger.error(str(e))

                if attempt == settings.max_retries - 1:
                    break

                await sleep(settings.retry_delay)

        #
        # 6. Fallback
        #

        app_logger.warning(
            "Primary provider failed. Using Gemini fallback."
        )

        fallback = self.providers[Provider.GEMINI]

        response = await fallback.generate(
            prompt=final_prompt,
        )

        #
        # Save caches
        #

        self.exact_cache.set(
            prompt,
            response,
        )

        await self.semantic_cache.set(
            prompt,
            response,
        )

        return response