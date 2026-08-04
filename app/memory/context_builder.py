from __future__ import annotations

from dataclasses import dataclass

from app.memory.service import MemoryService
from app.prompts.manager import PromptManager


@dataclass(frozen=True)
class PromptContext:
    """
    Normalized context sections ready for template rendering.
    """

    history: str
    memory: str
    documents: str
    user_input: str


class ContextBuilder:
    """
    Builds optimized LLM prompts from multiple context sources.

    Combines:
    - Conversation history (short-term memory)
    - Long-term memory facts (optional, Phase 3)
    - Retrieved documents (optional, Phase 7 RAG)
    - Current user input

    Empty optional sections are omitted so prompts stay clean.
    """

    def __init__(
        self,
        memory: MemoryService,
        prompt_manager: PromptManager | None = None,
    ) -> None:
        self.memory = memory
        self.prompt_manager = prompt_manager or PromptManager()

    async def build(
        self,
        *,
        session_id: str,
        user_input: str,
        prompt_name: str,
        version: int | None = None,
        long_term_memory: list[str] | None = None,
        documents: list[str] | None = None,
    ) -> str:
        context = await self.assemble(
            session_id=session_id,
            user_input=user_input,
            long_term_memory=long_term_memory,
            documents=documents,
        )

        return self.prompt_manager.render(
            prompt_name,
            version=version,
            history=context.history,
            memory=context.memory,
            documents=context.documents,
            input=context.user_input,
        )

    async def assemble(
        self,
        *,
        session_id: str,
        user_input: str,
        long_term_memory: list[str] | None = None,
        documents: list[str] | None = None,
    ) -> PromptContext:
        history = await self.memory.build_history_text(session_id)

        return PromptContext(
            history=history,
            memory=self._format_memory(long_term_memory),
            documents=self._format_documents(documents),
            user_input=user_input,
        )

    def _format_memory(self, facts: list[str] | None) -> str:
        if not facts:
            return ""

        cleaned = [fact.strip() for fact in facts if fact.strip()]
        if not cleaned:
            return ""

        lines = "\n".join(f"- {fact}" for fact in cleaned)
        return f"### Long-term memory\n{lines}\n\n"

    def _format_documents(self, documents: list[str] | None) -> str:
        if not documents:
            return ""

        cleaned = [doc.strip() for doc in documents if doc.strip()]
        if not cleaned:
            return ""

        blocks: list[str] = []
        for index, doc in enumerate(cleaned, start=1):
            blocks.append(f"[Document {index}]\n{doc}")

        body = "\n\n".join(blocks)
        return f"### Retrieved context\n{body}\n\n"
