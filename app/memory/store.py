from __future__ import annotations

import asyncio
from collections import defaultdict

from app.memory.models import Message
from app.memory.types import ConversationStore


class InMemoryConversationStore:
    """
    Async in-memory conversation store.

    Safe for concurrent requests within a single process.
    Replace with Redis/PostgreSQL by implementing ConversationStore.
    """

    def __init__(
        self,
        max_messages: int | None = None,
    ) -> None:
        self._sessions: dict[str, list[Message]] = defaultdict(list)
        self._max_messages = max_messages
        self._lock = asyncio.Lock()

    async def save_message(
        self,
        session_id: str,
        message: Message,
    ) -> None:
        async with self._lock:
            messages = self._sessions[session_id]
            messages.append(message)

            if self._max_messages is not None and len(messages) > self._max_messages:
                self._sessions[session_id] = messages[-self._max_messages :]

    async def get_messages(
        self,
        session_id: str,
    ) -> list[Message]:
        async with self._lock:
            return list(self._sessions.get(session_id, []))

    async def clear(
        self,
        session_id: str,
    ) -> None:
        async with self._lock:
            self._sessions[session_id] = []

    async def delete_session(
        self,
        session_id: str,
    ) -> None:
        async with self._lock:
            self._sessions.pop(session_id, None)


# Structural check that the in-memory store matches the Protocol.
_: type[ConversationStore] = InMemoryConversationStore
