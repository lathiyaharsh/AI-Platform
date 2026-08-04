from __future__ import annotations

from app.config.settings import settings
from app.memory.models import Message, MessageRole
from app.memory.store import InMemoryConversationStore
from app.memory.types import ConversationStore


class MemoryService:
    """
    Short-term conversation memory.

    Owns sliding-window history and history text formatting.
    Storage is injected so Redis/PostgreSQL can replace the in-memory store
    without Gateway changes.
    """

    def __init__(
        self,
        store: ConversationStore | None = None,
        window: int | None = None,
    ) -> None:
        self.window = window if window is not None else settings.memory_window
        self.store = store or InMemoryConversationStore(
            max_messages=self.window,
        )

    async def add_user_message(
        self,
        session_id: str,
        content: str,
    ) -> Message:
        message = Message(
            role=MessageRole.USER,
            content=content,
        )
        await self.store.save_message(session_id, message)
        return message

    async def add_assistant_message(
        self,
        session_id: str,
        content: str,
    ) -> Message:
        message = Message(
            role=MessageRole.ASSISTANT,
            content=content,
        )
        await self.store.save_message(session_id, message)
        return message

    async def get_history(
        self,
        session_id: str,
    ) -> list[Message]:
        messages = await self.store.get_messages(session_id)

        if len(messages) <= self.window:
            return messages

        return messages[-self.window :]

    async def build_history_text(
        self,
        session_id: str,
    ) -> str:
        """
        Format previous messages for prompt rendering.

        Does not include the current user turn.
        """
        messages = await self.get_history(session_id)

        if not messages:
            return ""

        parts: list[str] = []

        for message in messages:
            if message.role == MessageRole.USER:
                label = "User"
            else:
                label = "Assistant"
            parts.append(f"{label}: {message.content}")

        return "\n\n".join(parts) + "\n\n"

    async def clear(
        self,
        session_id: str,
    ) -> None:
        await self.store.clear(session_id)

    async def delete_session(
        self,
        session_id: str,
    ) -> None:
        await self.store.delete_session(session_id)
