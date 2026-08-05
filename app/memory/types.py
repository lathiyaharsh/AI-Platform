from typing import Protocol, runtime_checkable

from app.memory.models import MemoryFact, Message


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


@runtime_checkable
class LongTermMemoryStore(Protocol):
    """
    Storage interface for durable user facts.

    Swap in Redis/PostgreSQL later without changing LongTermMemoryService.
    """

    async def save_fact(
        self,
        fact: MemoryFact,
    ) -> None: ...

    async def get_facts(
        self,
        user_id: str,
    ) -> list[MemoryFact]: ...

    async def delete_fact(
        self,
        user_id: str,
        fact_id: str,
    ) -> bool: ...

    async def clear(
        self,
        user_id: str,
    ) -> None: ...
