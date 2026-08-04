from asyncio import sleep
from uuid import uuid4

from app.cache.exact_cache import ExactCache
from app.cache.semantic_cache import SemanticCache
from app.config.constants import Provider
from app.config.settings import settings
from app.gateway.providers.exceptions import ProviderError
from app.gateway.providers.gemini_provider import GeminiProvider
from app.gateway.providers.groq_provider import GroqProvider
from app.gateway.router import Router
from app.memory.context_builder import ContextBuilder
from app.memory.service import MemoryService
from app.observability.logger import app_logger
from app.prompts.manager import PromptManager


class Gateway:
    """
    Central AI Gateway.

    Responsibilities

    - Routing
    - Context building
    - Conversation memory
    - Exact cache
    - Semantic cache
    - Retry
    - Fallback
    """

    def __init__(
        self,
        memory: MemoryService | None = None,
    ) -> None:

        self.router = Router()

        self.prompt_manager = PromptManager()

        self.exact_cache = ExactCache()

        self.semantic_cache = SemanticCache()

        self.memory = memory or MemoryService()

        self.context_builder = ContextBuilder(
            memory=self.memory,
            prompt_manager=self.prompt_manager,
        )

        self.providers = {
            Provider.GROQ: GroqProvider(),
            Provider.GEMINI: GeminiProvider(),
        }

    async def generate(
        self,
        prompt: str,
        session_id: str | None = None,
    ) -> str:

        session_id = session_id or str(uuid4())

        #
        # 1. Intelligent Routing
        #

        decision = self.router.route(prompt)

        provider = self.providers[decision.provider]

        #
        # 2. Context Building (history + memory + docs + input)
        #

        final_prompt = await self.context_builder.build(
            session_id=session_id,
            user_input=prompt,
            prompt_name=decision.route.value,
        )

        #
        # 3. Exact Cache (keyed on full rendered prompt)
        #

        cached = self.exact_cache.get(final_prompt)

        if cached:

            app_logger.info("Exact Cache HIT")

            await self._persist_turn(
                session_id=session_id,
                prompt=prompt,
                response=cached,
            )

            return cached

        #
        # 4. Semantic Cache
        #

        cached = await self.semantic_cache.get(final_prompt)

        if cached:

            app_logger.info("Semantic Cache HIT")

            self.exact_cache.set(final_prompt, cached)

            await self._persist_turn(
                session_id=session_id,
                prompt=prompt,
                response=cached,
            )

            return cached

        #
        # 5. Retry
        #

        for attempt in range(settings.max_retries):

            try:

                app_logger.info(
                    f"Provider={decision.provider.value} "
                    f"Attempt={attempt + 1}"
                )

                response = await provider.generate(
                    prompt=final_prompt,
                )

                await self._persist_turn(
                    session_id=session_id,
                    prompt=prompt,
                    response=response,
                )

                self.exact_cache.set(
                    final_prompt,
                    response,
                )

                await self.semantic_cache.set(
                    final_prompt,
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

        await self._persist_turn(
            session_id=session_id,
            prompt=prompt,
            response=response,
        )

        self.exact_cache.set(
            final_prompt,
            response,
        )

        await self.semantic_cache.set(
            final_prompt,
            response,
        )

        return response

    async def _persist_turn(
        self,
        session_id: str,
        prompt: str,
        response: str,
    ) -> None:
        await self.memory.add_user_message(session_id, prompt)
        await self.memory.add_assistant_message(session_id, response)
