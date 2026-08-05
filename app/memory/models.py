from datetime import datetime, timezone
from enum import Enum
from uuid import uuid4

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


class MemoryFact(BaseModel):
    """
    Durable user preference / fact that survives conversation windows.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    content: str
    timestamp: datetime = Field(default_factory=utc_now)
