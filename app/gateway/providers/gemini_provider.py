from google import genai

from app.config.settings import settings
from app.gateway.providers.base import BaseProvider
from app.gateway.providers.exceptions import ProviderError


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

        try:
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=full_prompt,
            )

            text = response.text
            if text is None:
                raise ProviderError("Gemini returned empty response")
            return text
        except ProviderError:
            raise
        except Exception as e:
            raise ProviderError(str(e)) from e