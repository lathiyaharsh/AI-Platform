from typing import Protocol, runtime_checkable

from app.memory.models import Message


@runtime_checkable
class ConversationStore(Protocol):
    """
    Storage interface for conversation memory.

    Implementations can be swapped (in-memory, Redis, PostgreSQL)
    without changing MemoryService or Gateway.
    """

    async def save_message(
        self,
        session_id: str,
        message: Message,
    ) -> None: ...

    async def get_messages(
        self,
        session_id: str,
    ) -> list[Message]: ...

    async def clear(
        self,
        session_id: str,
    ) -> None: ...

    async def delete_session(
        self,
        session_id: str,
    ) -> None: ...
