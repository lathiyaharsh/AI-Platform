from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from app.rag.types import ChunkStrategy, DocumentType, RagErrorCode


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class DocumentMeta(BaseModel):
    """Catalog entry for an ingested document."""

    document_id: str
    filename: str
    source: str
    document_type: DocumentType
    chunk_count: int = 0
    char_count: int = 0
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class LoadedPage(BaseModel):
    """One logical page/section from a loader."""

    page: int | None = None
    text: str


class LoadedDocument(BaseModel):
    """Normalized loader output before chunking."""

    document_id: str
    filename: str
    source: str
    document_type: DocumentType
    pages: list[LoadedPage] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages if p.text.strip())


class Chunk(BaseModel):
    """A text chunk ready for embedding + storage."""

    chunk_id: str
    document_id: str
    text: str
    index: int = 0
    page: int | None = None
    source: str = ""
    filename: str = ""
    document_type: DocumentType = DocumentType.UNKNOWN
    strategy: ChunkStrategy = ChunkStrategy.SENTENCE
    metadata: dict[str, Any] = Field(default_factory=dict)


class VectorRecord(BaseModel):
    """One stored vector + payload."""

    id: str
    document_id: str
    chunk_id: str
    text: str
    embedding: list[float]
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchHit(BaseModel):
    """Single vector-search result."""

    document_id: str
    chunk_id: str
    text: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)
    page: int | None = None
    source: str = ""
    filename: str = ""


class IngestResult(BaseModel):
    """Outcome of document ingestion."""

    success: bool
    document: DocumentMeta | None = None
    chunks_created: int = 0
    upload_ms: float = 0.0
    embed_ms: float = 0.0
    store_ms: float = 0.0
    error: str | None = None
    error_code: RagErrorCode | None = None


class RetrievalResult(BaseModel):
    """Outcome of a retrieval query."""

    success: bool
    query: str
    hits: list[SearchHit] = Field(default_factory=list)
    top_k: int = 0
    retrieval_ms: float = 0.0
    embed_ms: float = 0.0
    context_chars: int = 0
    context_text: str = ""
    error: str | None = None
    error_code: RagErrorCode | None = None

    @property
    def scores(self) -> list[float]:
        return [h.score for h in self.hits]


class ReindexResult(BaseModel):
    """Outcome of reindexing one or all documents."""

    success: bool
    reindexed: int = 0
    failed: int = 0
    document_ids: list[str] = Field(default_factory=list)
    error: str | None = None
    error_code: RagErrorCode | None = None
