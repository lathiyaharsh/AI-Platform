from google import genai

from app.config.settings import settings
from app.gateway.providers.base import BaseProvider


class GeminiProvider(BaseProvider):
    def __init__(self):
        self.client = genai.Client(api_key=settings.gemini_api_key)

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> str:

        full_prompt = prompt

        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=full_prompt,
        )

        return response.text