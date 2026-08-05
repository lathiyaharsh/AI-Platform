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
from app.memory.long_term import LongTermMemoryService
from app.memory.service import MemoryService
from app.observability.logger import app_logger
from app.prompts.manager import PromptManager
from app.reflection.service import ReflectionService
from app.tools.service import ToolService


class Gateway:
    """
    Central AI Gateway.

    Responsibilities

    - Routing
    - Context building
    - Conversation memory
    - Long-term memory
    - Exact cache
    - Semantic cache
    - Retry
    - Fallback
    - Tool execution orchestration
    - Reflection orchestration
    """

    def __init__(
        self,
        memory: MemoryService | None = None,
        long_term: LongTermMemoryService | None = None,
        reflection: ReflectionService | None = None,
        tools: ToolService | None = None,
    ) -> None:

        self.router = Router()

        self.prompt_manager = PromptManager()

        self.exact_cache = ExactCache()

        self.semantic_cache = SemanticCache()

        self.memory = memory or MemoryService()

        self.long_term = long_term or LongTermMemoryService()

        self.context_builder = ContextBuilder(
            memory=self.memory,
            prompt_manager=self.prompt_manager,
            long_term=self.long_term,
        )

        self.providers = {
            Provider.GROQ: GroqProvider(),
            Provider.GEMINI: GeminiProvider(),
        }

        self.reflection = reflection or ReflectionService(
            providers=self.providers,
        )

        self.tools = tools or ToolService(
            providers=self.providers,
        )
        # Shared DI instances may be constructed before providers exist.
        if self.tools.providers is None:
            self.tools.providers = self.providers

    async def generate(
        self,
        prompt: str,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> str:

        session_id = session_id or str(uuid4())
        user_id = user_id or session_id

        #
        # 1. Intelligent Routing
        #

        decision = self.router.route(prompt)

        provider = self.providers[decision.provider]

        #
        # 2. Long-term memory extraction (explicit remember signals)
        #

        await self.long_term.extract_and_store(user_id, prompt)

        #
        # 3. Context Building (history + memory + docs + input)
        #

        built = await self.context_builder.build(
            session_id=session_id,
            user_id=user_id,
            user_input=prompt,
            prompt_name=decision.route.value,
        )
        final_prompt = built.text

        #
        # 4. Exact Cache (keyed on full rendered prompt)
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
        # 5. Semantic Cache
        #    Skip when personalized — shared template/memory text causes
        #    false hits across different user questions in the same session.
        #

        if built.personalized:
            app_logger.info("Semantic Cache SKIP (personalized context)")
        else:
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
        # 6. Retry (provider + tool loop)
        #

        for attempt in range(settings.max_retries):

            try:

                app_logger.info(
                    f"Provider={decision.provider.value} "
                    f"Attempt={attempt + 1}"
                )

                tool_result = await self.tools.run(
                    question=prompt,
                    context_prompt=final_prompt,
                    caller=provider,
                )
                response = tool_result.final_response
                used_tools = tool_result.used_tools

                response = await self._finalize_response(
                    question=prompt,
                    response=response,
                )

                await self._persist_turn(
                    session_id=session_id,
                    prompt=prompt,
                    response=response,
                )

                # Do not cache live tool answers (weather, datetime, …).
                if not used_tools:
                    self.exact_cache.set(
                        final_prompt,
                        response,
                    )

                    if not built.personalized:
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
        # 7. Fallback
        #

        app_logger.warning(
            "Primary provider failed. Using Gemini fallback."
        )

        fallback = self.providers[Provider.GEMINI]

        tool_result = await self.tools.run(
            question=prompt,
            context_prompt=final_prompt,
            caller=fallback,
        )
        response = tool_result.final_response
        used_tools = tool_result.used_tools

        response = await self._finalize_response(
            question=prompt,
            response=response,
        )

        await self._persist_turn(
            session_id=session_id,
            prompt=prompt,
            response=response,
        )

        if not used_tools:
            self.exact_cache.set(
                final_prompt,
                response,
            )

            if not built.personalized:
                await self.semantic_cache.set(
                    final_prompt,
                    response,
                )

        return response

    async def _finalize_response(
        self,
        question: str,
        response: str,
    ) -> str:
        """
        Run reflection after generation; Gateway only orchestrates.
        """
        result = await self.reflection.reflect(
            question=question,
            answer=response,
        )
        return result.final_response

    async def _persist_turn(
        self,
        session_id: str,
        prompt: str,
        response: str,
    ) -> None:
        await self.memory.add_user_message(session_id, prompt)
        await self.memory.add_assistant_message(session_id, response)
