import httpx

from app.config.settings import settings
from app.embeddings.base import BaseEmbedder


class HuggingFaceEmbedder(BaseEmbedder):

    def __init__(self):
        self.url = (
            f"https://api-inference.huggingface.co/"
            f"pipeline/feature-extraction/"
            f"{settings.hf_embed_model}"
        )

        self.headers = {
            "Authorization":
            f"Bearer {settings.huggingface_api_key}"
        }

    async def embed(
        self,
        text: str,
    ) -> list[float]:

        async with httpx.AsyncClient() as client:

            response = await client.post(
                self.url,
                headers=self.headers,
                json={
                    "inputs": text
                },
                timeout=60,
            )

            response.raise_for_status()

            vector = response.json()

            if isinstance(vector[0], list):
                return vector[0]

            return vector