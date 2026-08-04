from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Message(BaseModel):
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=utc_now)


class Conversation(BaseModel):
    session_id: str
    messages: list[Message] = Field(default_factory=list)
