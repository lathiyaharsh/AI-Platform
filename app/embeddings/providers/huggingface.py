import httpx

from app.config.settings import settings
from app.embeddings.base import BaseEmbedder


class HuggingFaceEmbedder(BaseEmbedder):
    """
    Hugging Face Embedding Provider
    Uses the Hugging Face Inference API.
    """

    def __init__(self):
        self.url = (
            "https://router.huggingface.co/hf-inference/models/"
            f"{settings.hf_embed_model}"
        )

        self.headers = {
            "Authorization": f"Bearer {settings.huggingface_api_key}",
            "Content-Type": "application/json",
        }

    async def embed(self, text: str) -> list[float]:

        payload = {
            "inputs": text,
        }

        async with httpx.AsyncClient(timeout=60) as client:

            response = await client.post(
                self.url,
                headers=self.headers,
                json=payload,
            )

            response.raise_for_status()

            data = response.json()

        # Some models return [[...]]
        if isinstance(data, list):
            if data and isinstance(data[0], list):
                return data[0]
            return data

        raise ValueError(f"Unexpected HF response: {data}")