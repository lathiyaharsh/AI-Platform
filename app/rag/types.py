from enum import Enum
from typing import Any, Protocol, runtime_checkable


class DocumentType(str, Enum):
    """
    Supported and future document formats.

    Loaders register against these values so new formats
    (docx, html, csv, …) plug in without changing RAGService.
    """

    TXT = "txt"
    MD = "md"
    PDF = "pdf"
    DOCX = "docx"
    HTML = "html"
    CSV = "csv"
    UNKNOWN = "unknown"


class ChunkStrategy(str, Enum):
    """Chunking strategies supported by the document pipeline."""

    FIXED = "fixed"
    SENTENCE = "sentence"


class VectorStoreBackend(str, Enum):
    """
    Vector store backends.

    Only MEMORY and PGVECTOR are implemented now; other values reserve the
    settings contract for Qdrant, Pinecone, etc.
    """

    MEMORY = "memory"
    PGVECTOR = "pgvector"
    QDRANT = "qdrant"
    PINECONE = "pinecone"
    WEAVIATE = "weaviate"
    MILVUS = "milvus"
    FAISS = "faiss"


class RagErrorCode(str, Enum):
    """Standardized RAG failure codes for API and logging."""

    UNSUPPORTED_DOCUMENT = "unsupported_document"
    CORRUPTED_DOCUMENT = "corrupted_document"
    EMBEDDING_FAILURE = "embedding_failure"
    RETRIEVAL_FAILURE = "retrieval_failure"
    STORAGE_FAILURE = "storage_failure"
    DOCUMENT_NOT_FOUND = "document_not_found"
    DISABLED = "disabled"
    UNKNOWN = "unknown"


@runtime_checkable
class VectorStore(Protocol):
    """
    Vector storage contract.

    Implementations: MemoryVectorStore, PgVectorStore;
    Qdrant / Pinecone / Weaviate / Milvus / FAISS later
    without changing RAGService or Retriever.
    """

    async def add(
        self,
        *,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None: ...

    async def search(
        self,
        embedding: list[float],
        *,
        top_k: int = 5,
        min_similarity: float = 0.0,
        filters: dict[str, Any] | None = None,
    ) -> list[Any]: ...

    async def delete(self, document_id: str) -> int: ...

    async def clear(self) -> None: ...


@runtime_checkable
class Reranker(Protocol):
    """
    Rerank retrieved candidates.

    SimilarityReranker now; Cross-Encoder / Cohere / Jina later
    without changing Retriever.
    """

    async def rerank(
        self,
        query: str,
        results: list[Any],
        *,
        top_k: int | None = None,
    ) -> list[Any]: ...


@runtime_checkable
class DocumentLoader(Protocol):
    """Load raw bytes/text into a normalized LoadedDocument."""

    supported_types: frozenset[DocumentType]

    async def load(
        self,
        *,
        content: bytes,
        filename: str,
        document_id: str,
        source: str | None = None,
    ) -> Any: ...
