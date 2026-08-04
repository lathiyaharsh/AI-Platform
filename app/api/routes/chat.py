from fastapi import APIRouter, Depends

from app.api.dependencies.gateway import get_gateway
from app.gateway.gateway import Gateway
from app.prompts.manager import PromptManager

router = APIRouter()


@router.get("/chat")
async def chat(
    gateway: Gateway = Depends(get_gateway),
):

    answer = await gateway.generate(
        prompt="Explain FastAPI in one sentence."
    )

    return {
        "response": answer
    }