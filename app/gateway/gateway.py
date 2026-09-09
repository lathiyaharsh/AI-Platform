from asyncio import sleep
from time import perf_counter
from uuid import uuid4

from app.cache.exact_cache import ExactCache
from app.cache.semantic_cache import SemanticCache
from app.config.constants import Provider, RouteType
from app.config.settings import settings
from app.gateway.providers.exceptions import ProviderError
from app.gateway.providers.gemini_provider import GeminiProvider
from app.gateway.providers.groq_provider import GroqProvider
from app.gateway.router import Router
from app.memory.context_builder import ContextBuilder
from app.memory.long_term import LongTermMemoryService
from app.memory.service import MemoryService
from app.observability.logger import app_logger
from app.observability.models import ToolCallMetric
from app.observability.service import ObservabilityService
from app.observability.types import CacheKind, RequestStatus
from app.prompts.manager import PromptManager
from app.rag.service import RAGService
from app.reflection.service import ReflectionService
from app.tools.models import ToolLoopResult
from app.tools.service import ToolService


class Gateway:
    """
    Central AI Gateway.

    Responsibilities

    - Routing
    - Context building
    - Conversation memory
    - Long-term memory
    - RAG orchestration (via RAGService)
    - Exact cache
    - Semantic cache
    - Retry
    - Fallback
    - Tool execution orchestration
    - Reflection orchestration
    - Observability orchestration (via ObservabilityService)
    """

    def __init__(
        self,
        memory: MemoryService | None = None,
        long_term: LongTermMemoryService | None = None,
        reflection: ReflectionService | None = None,
        tools: ToolService | None = None,
        rag: RAGService | None = None,
        observability: ObservabilityService | None = None,
    ) -> None:

        self.router = Router()

        self.prompt_manager = PromptManager()

        self.exact_cache = ExactCache()

        self.semantic_cache = SemanticCache()

        self.memory = memory or MemoryService()

        self.long_term = long_term or LongTermMemoryService()

        self.rag = rag or RAGService()

        self.obs = observability or ObservabilityService()

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
        *,
        force_rag: bool = False,
        top_k: int | None = None,
        min_similarity: float | None = None,
    ) -> str:

        session_id = session_id or str(uuid4())
        user_id = user_id or session_id

        request_id = await self.obs.start_request(
            session_id=session_id,
            user_id=user_id,
            prompt=prompt,
        )
        status = RequestStatus.SUCCESS
        response: str | None = None
        final_prompt = prompt

        try:
            #
            # 1. Intelligent Routing
            #

            route_started = perf_counter()
            decision = self.router.route(
                prompt,
                force_rag=force_rag,
                rag_available=self.rag.enabled and self.rag.has_documents,
            )
            await self.obs.record_routing(
                request_id,
                route=decision.route.value,
                provider=decision.provider.value,
                model=decision.model,
                duration_ms=(perf_counter() - route_started) * 1000,
            )

            provider = self.providers[decision.provider]

            #
            # 2. Long-term memory extraction (explicit remember signals)
            #

            await self.long_term.extract_and_store(user_id, prompt)

            #
            # 3. RAG retrieval (Gateway only calls RAGService)
            #

            documents: list[str] | None = None
            if (
                settings.enable_rag
                and self.rag.enabled
                and (
                    force_rag
                    or decision.route == RouteType.RAG
                )
            ):
                retrieval = await self.rag.retrieve(
                    prompt,
                    top_k=top_k,
                    min_similarity=min_similarity,
                )
                await self.obs.record_rag(
                    request_id,
                    documents_retrieved=len(
                        {h.document_id for h in retrieval.hits}
                    ),
                    chunk_count=len(retrieval.hits),
                    similarity_scores=retrieval.scores,
                    retrieval_time_ms=retrieval.retrieval_ms,
                    embedding_time_ms=retrieval.embed_ms,
                    context_size=retrieval.context_chars,
                    success=retrieval.success,
                    error=retrieval.error,
                )
                if retrieval.success and retrieval.hits:
                    documents = self.rag.documents_for_context(retrieval)
                    app_logger.info(
                        f"RAG context attached sources={len(documents)} "
                        f"context_chars={retrieval.context_chars}"
                    )
                elif not retrieval.success:
                    app_logger.warning(
                        f"RAG retrieval skipped: {retrieval.error}"
                    )

            #
            # 4. Context Building (history + memory + docs + tools + input)
            #

            history_before = await self.memory.get_history(session_id)
            prompt_name = decision.route.value
            prompt_version = self.prompt_manager.get_version(prompt_name).version
            await self.obs.record_prompt(
                request_id,
                prompt_name=prompt_name,
                prompt_version=prompt_version,
            )

            built = await self.context_builder.build(
                session_id=session_id,
                user_id=user_id,
                user_input=prompt,
                prompt_name=prompt_name,
                documents=documents,
            )
            final_prompt = built.text

            #
            # 5. Exact Cache (keyed on full rendered prompt)
            #

            cached = self.exact_cache.get(final_prompt)

            if cached:

                app_logger.info("Exact Cache HIT")
                await self.obs.record_cache(
                    request_id,
                    kind=CacheKind.EXACT,
                    hit=True,
                )

                await self._persist_turn(
                    session_id=session_id,
                    prompt=prompt,
                    response=cached,
                )
                await self._record_memory_metrics(
                    request_id=request_id,
                    session_id=session_id,
                    messages_loaded=len(history_before),
                )

                status = RequestStatus.CACHE_HIT
                response = cached
                return cached

            await self.obs.record_cache(
                request_id,
                kind=CacheKind.EXACT,
                hit=False,
            )

            #
            # 6. Semantic Cache
            #    Skip when personalized — shared template/memory text causes
            #    false hits across different user questions in the same session.
            #

            if built.personalized:
                app_logger.info("Semantic Cache SKIP (personalized context)")
                await self.obs.record_cache(
                    request_id,
                    kind=CacheKind.SEMANTIC,
                    hit=False,
                    skipped=True,
                )
            else:
                cached = await self.semantic_cache.get(final_prompt)

                if cached:

                    app_logger.info("Semantic Cache HIT")
                    await self.obs.record_cache(
                        request_id,
                        kind=CacheKind.SEMANTIC,
                        hit=True,
                    )

                    self.exact_cache.set(final_prompt, cached)

                    await self._persist_turn(
                        session_id=session_id,
                        prompt=prompt,
                        response=cached,
                    )
                    await self._record_memory_metrics(
                        request_id=request_id,
                        session_id=session_id,
                        messages_loaded=len(history_before),
                    )

                    status = RequestStatus.CACHE_HIT
                    response = cached
                    return cached

                await self.obs.record_cache(
                    request_id,
                    kind=CacheKind.SEMANTIC,
                    hit=False,
                )

            #
            # 7. Retry (provider + tool loop)
            #

            for attempt in range(settings.max_retries):

                try:

                    app_logger.info(
                        f"Provider={decision.provider.value} "
                        f"Route={decision.route.value} "
                        f"Attempt={attempt + 1}"
                    )

                    tool_result = await self.tools.run(
                        question=prompt,
                        context_prompt=final_prompt,
                        caller=provider,
                        model=decision.model,
                    )
                    response = tool_result.final_response
                    used_tools = tool_result.used_tools

                    await self._record_tool_metrics(
                        request_id,
                        tool_result,
                    )

                    response = await self._finalize_response(
                        request_id=request_id,
                        question=prompt,
                        response=response,
                    )

                    await self._persist_turn(
                        session_id=session_id,
                        prompt=prompt,
                        response=response,
                    )
                    await self._record_memory_metrics(
                        request_id=request_id,
                        session_id=session_id,
                        messages_loaded=len(history_before),
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
                    await self.obs.record_retry(
                        request_id,
                        attempt=attempt + 1,
                        error=str(e),
                    )

                    if attempt == settings.max_retries - 1:
                        break

                    await sleep(settings.retry_delay)

            #
            # 8. Fallback
            #

            app_logger.warning(
                "Primary provider failed. Using Gemini fallback."
            )

            fallback = self.providers[Provider.GEMINI]
            fallback_model = "gemini-2.5-flash"
            await self.obs.record_fallback(
                request_id,
                provider=Provider.GEMINI.value,
                model=fallback_model,
            )

            tool_result = await self.tools.run(
                question=prompt,
                context_prompt=final_prompt,
                caller=fallback,
                model=fallback_model,
            )
            response = tool_result.final_response
            used_tools = tool_result.used_tools

            await self._record_tool_metrics(
                request_id,
                tool_result,
            )

            response = await self._finalize_response(
                request_id=request_id,
                question=prompt,
                response=response,
            )

            await self._persist_turn(
                session_id=session_id,
                prompt=prompt,
                response=response,
            )
            await self._record_memory_metrics(
                request_id=request_id,
                session_id=session_id,
                messages_loaded=len(history_before),
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

            status = RequestStatus.FALLBACK
            return response

        except Exception as exc:
            status = RequestStatus.ERROR
            app_logger.error(f"Gateway failure: {exc}")
            raise

        finally:
            await self.obs.end_request(
                request_id,
                status=status,
                response=response,
                input_text=final_prompt,
                output_text=response,
            )

    async def _finalize_response(
        self,
        question: str,
        response: str,
        *,
        request_id: str | None = None,
    ) -> str:
        """
        Run reflection after generation; Gateway only orchestrates.
        """
        result = await self.reflection.reflect(
            question=question,
            answer=response,
        )
        if request_id is not None:
            await self.obs.record_reflection(
                request_id,
                enabled=self.reflection.enabled,
                used=not result.skipped,
                score=result.score,
                improvement_performed=result.improved,
                duration_ms=result.duration_ms,
                skipped=result.skipped,
                skip_reason=result.skip_reason,
                pass_count=len(result.passes),
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

    async def _record_memory_metrics(
        self,
        *,
        request_id: str,
        session_id: str,
        messages_loaded: int,
    ) -> None:
        await self.obs.record_memory(
            request_id,
            session_id=session_id,
            messages_loaded=messages_loaded,
            messages_stored=2,
            sliding_window_size=self.memory.window,
        )

    async def _record_tool_metrics(
        self,
        request_id: str,
        tool_result: ToolLoopResult,
    ) -> None:
        executions: list[ToolCallMetric] = []
        for round_item in tool_result.rounds:
            for execution in round_item.executions:
                executions.append(
                    ToolCallMetric(
                        tool_name=execution.tool,
                        execution_time_ms=execution.duration_ms,
                        arguments=execution.arguments,
                        success=execution.success,
                        error=execution.error,
                    )
                )

        await self.obs.record_tools(
            request_id,
            used_tools=tool_result.used_tools,
            duration_ms=tool_result.duration_ms,
            executions=executions,
            skipped=tool_result.skipped,
            skip_reason=tool_result.skip_reason,
        )
