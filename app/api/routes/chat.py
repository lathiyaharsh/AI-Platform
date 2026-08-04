from uuid import uuid4

from fastapi import APIRouter, Depends

from app.api.dependencies.gateway import get_gateway
from app.api.schemas.chat import ChatRequest, ChatResponse
from app.gateway.gateway import Gateway

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    gateway: Gateway = Depends(get_gateway),
):

    session_id = request.session_id or str(uuid4())

    answer = await gateway.generate(
        prompt=request.message,
        session_id=session_id,
    )

    return ChatResponse(
        session_id=session_id,
        response=answer,
    )
