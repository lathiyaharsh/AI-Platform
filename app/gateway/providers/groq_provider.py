from groq import AsyncGroq

from app.config.settings import settings
from app.gateway.providers.base import BaseProvider
from app.gateway.providers.exceptions import ProviderError


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

        try:
            response = await self.client.chat.completions.create(
                model=settings.default_model,
                messages=messages,
            )

            content = response.choices[0].message.content
            if content is None:
                raise ProviderError("Groq returned empty response")
            return content

        except ProviderError:
            raise
        except Exception as e:
            raise ProviderError(str(e)) from e