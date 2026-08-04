from __future__ import annotations

from dataclasses import dataclass

from app.config.settings import settings
from app.embeddings.service import EmbeddingService
from app.utils.similarity import cosine_similarity


@dataclass
class SemanticCacheItem:
    prompt: str
    embedding: list[float]
    response: str


class SemanticCache:
    """
    Simple in-memory semantic cache.

    Later this can be replaced with:
    - Redis
    - pgvector
    - Pinecone
    - Qdrant
    """

    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.items: list[SemanticCacheItem] = []

    async def get(self, prompt: str) -> str | None:
        """
        Returns cached response if a similar prompt exists.
        """

        if not self.items:
            return None

        query_embedding = await self.embedding_service.embed(prompt)

        best_score = 0.0
        best_item = None

        for item in self.items:

            score = cosine_similarity(
                query_embedding,
                item.embedding,
            )

            if score > best_score:
                best_score = score
                best_item = item

        if (
            best_item
            and best_score >= settings.semantic_cache_threshold
        ):
            return best_item.response

        return None

    async def set(
        self,
        prompt: str,
        response: str,
    ) -> None:

        embedding = await self.embedding_service.embed(prompt)

        self.items.append(
            SemanticCacheItem(
                prompt=prompt,
                embedding=embedding,
                response=response,
            )
        )   