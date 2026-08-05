from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    session_id: str | None = None
    user_id: str | None = None


class ChatResponse(BaseModel):
    session_id: str
    response: str
