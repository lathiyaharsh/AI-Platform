from app.embeddings.providers.huggingface import (
    HuggingFaceEmbedder,
)


class EmbeddingService:
    """
    Platform embedding facade.

    Reused by SemanticCache and RAG — never duplicate provider logic.
    """

    def __init__(self) -> None:
        self.provider = HuggingFaceEmbedder()

    async def embed(self, text: str) -> list[float]:
        return await self.provider.embed(text)

    async def embed_many(self, texts: list[str]) -> list[list[float]]:
        """
        Embed multiple texts via the same provider.

        Sequential by design so we do not fork embedding logic;
        batch APIs can replace this later without callers changing.
        """
        vectors: list[list[float]] = []
        for text in texts:
            vectors.append(await self.embed(text))
        return vectors
