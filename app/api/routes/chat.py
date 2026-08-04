from fastapi import APIRouter

from app.gateway.providers.groq_provider import GroqProvider

router = APIRouter()


@router.get("/chat")
async def chat():

    provider = GroqProvider()

    answer = await provider.generate(
        "Say hello in one sentence."
    )

    return {
        "response": answer,
    }