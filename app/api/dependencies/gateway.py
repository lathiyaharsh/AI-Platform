from functools import lru_cache

from app.embeddings.service import EmbeddingService
from app.gateway.gateway import Gateway
from app.memory.long_term import LongTermMemoryService
from app.memory.service import MemoryService
from app.rag.service import RAGService, create_vector_store
from app.tools.service import ToolService


@lru_cache
def get_memory_service() -> MemoryService:
    return MemoryService()


@lru_cache
def get_long_term_memory() -> LongTermMemoryService:
    return LongTermMemoryService()


@lru_cache
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()


@lru_cache
def get_rag_service() -> RAGService:
    """
    Shared RAGService for DI.

    Embeddings and vector store are created once so upload + chat
    share the same in-memory catalog and vectors.
    """
    embeddings = get_embedding_service()
    store = create_vector_store()
    return RAGService(
        embeddings=embeddings,
        store=store,
    )


@lru_cache
def get_tool_service() -> ToolService:
    """
    Shared ToolService for DI.

    Providers are injected by Gateway after construction so the
    tool loop reuses the same Groq/Gemini instances.
    """
    return ToolService(tools=None)


@lru_cache
def get_gateway() -> Gateway:
    return Gateway(
        memory=get_memory_service(),
        long_term=get_long_term_memory(),
        tools=get_tool_service(),
        rag=get_rag_service(),
    )
