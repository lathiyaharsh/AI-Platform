from app.embeddings.providers.huggingface import (
    HuggingFaceEmbedder,
)


class EmbeddingService:

    def __init__(self):

        self.provider = HuggingFaceEmbedder()

    async def embed(
        self,
        text: str,
    ) -> list[float]:

        return await self.provider.embed(text)