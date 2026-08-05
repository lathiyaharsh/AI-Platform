from __future__ import annotations

from abc import ABC, abstractmethod

from app.rag.models import SearchHit


class BaseReranker(ABC):
    """
    Reranker abstraction.

    SimilarityReranker sorts by cosine score today.
    Future: CrossEncoderReranker, CohereReranker, JinaReranker.
    Retriever depends only on this ABC / Reranker protocol.
    """

    @abstractmethod
    async def rerank(
        self,
        query: str,
        results: list[SearchHit],
        *,
        top_k: int | None = None,
    ) -> list[SearchHit]:
        raise NotImplementedError


class SimilarityReranker(BaseReranker):
    """Sort candidates by similarity score (descending)."""

    async def rerank(
        self,
        query: str,
        results: list[SearchHit],
        *,
        top_k: int | None = None,
    ) -> list[SearchHit]:
        _ = query  # reserved for cross-encoder / LLM rerankers
        ordered = sorted(results, key=lambda hit: hit.score, reverse=True)
        if top_k is not None:
            return ordered[:top_k]
        return ordered
