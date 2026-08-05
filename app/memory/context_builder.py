from __future__ import annotations

from dataclasses import dataclass

from app.memory.long_term import LongTermMemoryService
from app.memory.service import MemoryService
from app.prompts.manager import PromptManager


@dataclass(frozen=True)
class PromptContext:
    """
    Normalized context sections ready for template rendering.

    Order in the final prompt (via templates):
    System → Memory → Retrieved Chunks → Tool Results → History → User
    """

    history: str
    memory: str
    documents: str
    tool_results: str
    user_input: str

    @property
    def is_personalized(self) -> bool:
        """True when prompt depends on session/user-specific context."""
        return bool(
            self.history.strip()
            or self.memory.strip()
            or self.documents.strip()
            or self.tool_results.strip()
        )


@dataclass(frozen=True)
class BuiltPrompt:
    text: str
    personalized: bool


class ContextBuilder:
    """
    Builds optimized LLM prompts from multiple context sources.

    Combines:
    - Conversation history (short-term memory)
    - Long-term memory facts
    - Retrieved document chunks (RAG)
    - Tool execution results (optional)
    - Current user input

    Empty optional sections are omitted so prompts stay clean.
    """

    def __init__(
        self,
        memory: MemoryService,
        prompt_manager: PromptManager | None = None,
        long_term: LongTermMemoryService | None = None,
    ) -> None:
        self.memory = memory
        self.prompt_manager = prompt_manager or PromptManager()
        self.long_term = long_term

    async def build(
        self,
        *,
        session_id: str,
        user_input: str,
        prompt_name: str,
        user_id: str | None = None,
        version: int | None = None,
        long_term_memory: list[str] | None = None,
        documents: list[str] | None = None,
        tool_results: list[str] | None = None,
    ) -> BuiltPrompt:
        context = await self.assemble(
            session_id=session_id,
            user_input=user_input,
            user_id=user_id,
            long_term_memory=long_term_memory,
            documents=documents,
            tool_results=tool_results,
        )

        text = self.prompt_manager.render(
            prompt_name,
            version=version,
            history=context.history,
            memory=context.memory,
            documents=context.documents,
            tool_results=context.tool_results,
            input=context.user_input,
        )

        return BuiltPrompt(
            text=text,
            personalized=context.is_personalized,
        )

    async def assemble(
        self,
        *,
        session_id: str,
        user_input: str,
        user_id: str | None = None,
        long_term_memory: list[str] | None = None,
        documents: list[str] | None = None,
        tool_results: list[str] | None = None,
    ) -> PromptContext:
        history = await self.memory.build_history_text(session_id)

        facts = long_term_memory
        if facts is None and self.long_term is not None:
            facts = await self.long_term.get_fact_texts(
                user_id or session_id,
            )

        return PromptContext(
            history=history,
            memory=self._format_memory(facts),
            documents=self._format_documents(documents),
            tool_results=self._format_tool_results(tool_results),
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

    def _format_tool_results(self, tool_results: list[str] | None) -> str:
        if not tool_results:
            return ""

        cleaned = [item.strip() for item in tool_results if item.strip()]
        if not cleaned:
            return ""

        blocks: list[str] = []
        for index, item in enumerate(cleaned, start=1):
            blocks.append(f"[Tool result {index}]\n{item}")

        body = "\n\n".join(blocks)
        return f"### Tool results\n{body}\n\n"
