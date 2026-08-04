from groq import AsyncGroq

from app.config.settings import settings
from app.gateway.providers.base import BaseProvider


class GroqProvider(BaseProvider):
    def __init__(self):
        self.client = AsyncGroq(api_key=settings.groq_api_key)

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> str:

        messages = []

        if system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": system_prompt,
                }
            )

        messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        response = await self.client.chat.completions.create(
            model=settings.default_model,
            messages=messages,
        )

        return response.choices[0].message.content