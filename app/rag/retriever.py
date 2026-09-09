from __future__ import annotations

import time
from typing import Any

from app.config.settings import settings
from app.embeddings.service import EmbeddingService
from app.observability.logger import app_logger
from app.rag.models import RetrievalResult, SearchHit
from app.rag.reranker import BaseReranker, SimilarityReranker
from app.rag.storage import MemoryVectorStore
from app.rag.types import RagErrorCode, Reranker, VectorStore


class Retriever:
    """
    Retrieve top chunks for a query.

    Flow
    1. Embed query via EmbeddingService
    2. Search VectorStore (cosine similarity)
    3. Rerank via injected Reranker
    4. Return top_k hits above min_similarity
    """

    def __init__(
        self,
        *,
        embeddings: EmbeddingService | None = None,
        store: VectorStore | None = None,
        reranker: Reranker | BaseReranker | None = None,
        top_k: int | None = None,
        min_similarity: float | None = None,
    ) -> None:
        self.embeddings = embeddings or EmbeddingService()
        self.store = store or MemoryVectorStore()
        self.reranker = reranker or SimilarityReranker()
        self.top_k = top_k if top_k is not None else settings.top_k
        self.min_similarity = (
            min_similarity
            if min_similarity is not None
            else settings.min_similarity
        )

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        min_similarity: float | None = None,
        filters: dict[str, Any] | None = None,
    ) -> RetrievalResult:
        k = top_k if top_k is not None else self.top_k
        threshold = (
            min_similarity
            if min_similarity is not None
            else self.min_similarity
        )
        started = time.perf_counter()
        embed_ms = 0.0

        try:
            embed_started = time.perf_counter()
            query_embedding = await self.embeddings.embed(query)
            embed_ms = (time.perf_counter() - embed_started) * 1000

            # Fetch a wider candidate set so rerankers can reshuffle.
            candidate_k = max(k * 3, k)
            hits: list[SearchHit] = await self.store.search(
                query_embedding,
                top_k=candidate_k,
                min_similarity=threshold,
                filters=filters,
            )

            # Soft fallback: short / typo queries often score just under
            # the hard threshold; still return the nearest chunks if close.
            if not hits:
                soft_floor = max(0.35, threshold - 0.15)
                probe = await self.store.search(
                    query_embedding,
                    top_k=candidate_k,
                    min_similarity=soft_floor,
                    filters=filters,
                )
                if probe:
                    app_logger.warning(
                        f"RAG retrieve soft-fallback "
                        f"threshold={threshold} soft_floor={soft_floor} "
                        f"best_score={probe[0].score:.4f} hits={len(probe)}"
                    )
                    hits = probe
                else:
                    await self._log_empty_retrieval(
                        query_embedding=query_embedding,
                        threshold=threshold,
                        filters=filters,
                    )

            ranked = await self.reranker.rerank(query, hits, top_k=k)
            retrieval_ms = (time.perf_counter() - started) * 1000

            app_logger.info(
                f"RAG retrieve top_k={k} min_similarity={threshold} "
                f"hits={len(ranked)} "
                f"scores={[round(h.score, 4) for h in ranked]} "
                f"embed_ms={embed_ms:.1f} retrieval_ms={retrieval_ms:.1f}"
            )

            return RetrievalResult(
                success=True,
                query=query,
                hits=ranked,
                top_k=k,
                retrieval_ms=retrieval_ms,
                embed_ms=embed_ms,
            )
        except Exception as exc:
            retrieval_ms = (time.perf_counter() - started) * 1000
            app_logger.error(f"RAG retrieval failure: {exc}")
            return RetrievalResult(
                success=False,
                query=query,
                top_k=k,
                retrieval_ms=retrieval_ms,
                embed_ms=embed_ms,
                error=str(exc),
                error_code=RagErrorCode.RETRIEVAL_FAILURE,
            )

    async def _log_empty_retrieval(
        self,
        *,
        query_embedding: list[float],
        threshold: float,
        filters: dict[str, Any] | None,
    ) -> None:
        """Explain zero-hit searches (empty store vs threshold too high)."""
        store_size = getattr(self.store, "size", None)
        if store_size == 0:
            app_logger.warning("RAG retrieve empty: vector store has 0 records")
            return

        try:
            best = await self.store.search(
                query_embedding,
                top_k=1,
                min_similarity=0.0,
                filters=filters,
            )
        except Exception as exc:
            app_logger.warning(
                f"RAG retrieve empty: could not probe best score ({exc})"
            )
            return

        if not best:
            app_logger.warning(
                f"RAG retrieve empty: store_size={store_size} "
                f"no candidates matched filters"
            )
            return

        app_logger.warning(
            f"RAG retrieve empty: best_score={best[0].score:.4f} "
            f"below min_similarity={threshold} "
            f"store_size={store_size} "
            f"chunk_id={best[0].chunk_id}"
        )
