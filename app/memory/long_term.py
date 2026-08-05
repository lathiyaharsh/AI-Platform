from __future__ import annotations

import asyncio
import re
from collections import defaultdict

from app.config.settings import settings
from app.memory.models import MemoryFact
from app.memory.types import LongTermMemoryStore

# Conservative patterns — only clear "remember / preference" signals.
_EXTRACTORS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(?i)^\s*remember(?:\s+that)?\s+my name is\s+(.+)$"), "name"),
    (re.compile(r"(?i)^\s*remember(?:\s+that)?[:\s]+(.+)$"), "raw"),
    (re.compile(r"(?i)^\s*please remember[:\s]+(.+)$"), "raw"),
    (re.compile(r"(?i)^\s*my name is\s+(.+)$"), "name"),
    (re.compile(r"(?i)^\s*call me\s+(.+)$"), "call_me"),
    (re.compile(r"(?i)^\s*i prefer\s+(.+)$"), "prefer"),
)


class InMemoryLongTermStore:
    """
    Async in-memory long-term fact store.

    Safe for concurrent requests within a single process.
    Replace with Redis/PostgreSQL by implementing LongTermMemoryStore.
    """

    def __init__(self) -> None:
        self._facts: dict[str, list[MemoryFact]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def save_fact(self, fact: MemoryFact) -> None:
        async with self._lock:
            self._facts[fact.user_id].append(fact)

    async def get_facts(self, user_id: str) -> list[MemoryFact]:
        async with self._lock:
            return list(self._facts.get(user_id, []))

    async def delete_fact(self, user_id: str, fact_id: str) -> bool:
        async with self._lock:
            facts = self._facts.get(user_id, [])
            kept = [fact for fact in facts if fact.id != fact_id]
            if len(kept) == len(facts):
                return False
            self._facts[user_id] = kept
            return True

    async def clear(self, user_id: str) -> None:
        async with self._lock:
            self._facts[user_id] = []


_: type[LongTermMemoryStore] = InMemoryLongTermStore


class LongTermMemoryService:
    """
    Durable memory of user facts / preferences.

    Survives short-term conversation window eviction.
    Until auth exists, Gateway may key facts by session_id.
    """

    def __init__(
        self,
        store: LongTermMemoryStore | None = None,
        max_facts: int | None = None,
    ) -> None:
        self.store = store or InMemoryLongTermStore()
        self.max_facts = (
            max_facts if max_facts is not None else settings.long_term_memory_max_facts
        )

    async def remember(self, user_id: str, content: str) -> MemoryFact | None:
        cleaned = content.strip()
        if not cleaned:
            return None

        existing = await self.get_fact_texts(user_id)
        if cleaned.lower() in {fact.lower() for fact in existing}:
            return None

        fact = MemoryFact(user_id=user_id, content=cleaned)
        await self.store.save_fact(fact)
        await self._enforce_limit(user_id)
        return fact

    async def get_facts(self, user_id: str) -> list[MemoryFact]:
        facts = await self.store.get_facts(user_id)
        if len(facts) <= self.max_facts:
            return facts
        return facts[-self.max_facts :]

    async def get_fact_texts(self, user_id: str) -> list[str]:
        return [fact.content for fact in await self.get_facts(user_id)]

    async def forget(self, user_id: str, fact_id: str) -> bool:
        return await self.store.delete_fact(user_id, fact_id)

    async def clear(self, user_id: str) -> None:
        await self.store.clear(user_id)

    async def extract_and_store(
        self,
        user_id: str,
        text: str,
    ) -> list[MemoryFact]:
        """
        Pull explicit memory signals from a user message and store them.
        """
        stored: list[MemoryFact] = []

        for pattern, kind in _EXTRACTORS:
            match = pattern.match(text.strip())
            if not match:
                continue

            content = match.group(1).strip().rstrip(".")
            if not content:
                continue

            if kind == "name":
                content = f"User's name is {content}"
            elif kind == "call_me":
                content = f"User prefers to be called {content}"
            elif kind == "prefer":
                content = f"User prefers {content}"

            fact = await self.remember(user_id, content)
            if fact is not None:
                stored.append(fact)
            break

        return stored

    async def _enforce_limit(self, user_id: str) -> None:
        facts = await self.store.get_facts(user_id)
        overflow = len(facts) - self.max_facts
        if overflow <= 0:
            return

        for stale in facts[:overflow]:
            await self.store.delete_fact(user_id, stale.id)
