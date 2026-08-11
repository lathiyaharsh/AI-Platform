from __future__ import annotations

from typing import Any

from app.config.settings import settings
from app.embeddings.service import EmbeddingService
from app.observability.logger import app_logger
from app.rag.chunking import Chunker
from app.rag.context import RagContextBuilder
from app.rag.models import (
    DocumentMeta,
    IngestResult,
    ReindexResult,
    RetrievalResult,
)
from app.rag.pipeline import DocumentPipeline
from app.rag.reranker import SimilarityReranker
from app.rag.retriever import Retriever
from app.rag.pgvector_store import PgVectorStore
from app.rag.storage import MemoryVectorStore
from app.rag.types import (
    RagErrorCode,
    Reranker,
    VectorStore,
    VectorStoreBackend,
)


def create_vector_store(
    backend: str | VectorStoreBackend | None = None,
) -> VectorStore:
    """
    Factory for vector backends.

    Supported: memory, pgvector. Other backends raise clearly so
    settings stay future-ready without silent misconfiguration.
    """
    value = backend or settings.vector_store
    if isinstance(value, str):
        value = VectorStoreBackend(value.lower())

    if value == VectorStoreBackend.MEMORY:
        return MemoryVectorStore()
    if value == VectorStoreBackend.PGVECTOR:
        return PgVectorStore()

    raise ValueError(
        f"Vector store backend '{value.value}' is not implemented yet. "
        "Use VECTOR_STORE=memory or VECTOR_STORE=pgvector."
    )


class RAGService:
    """
    High-level RAG API for Gateway and HTTP routes.

    Responsibilities
    - ingest / reindex / list / delete documents
    - retrieve context for a query
    - format attributed documents for ContextBuilder

    Gateway communicates only with RAGService.
    """

    def __init__(
        self,
        *,
        embeddings: EmbeddingService | None = None,
        store: VectorStore | None = None,
        pipeline: DocumentPipeline | None = None,
        retriever: Retriever | None = None,
        context_builder: RagContextBuilder | None = None,
        reranker: Reranker | None = None,
        enabled: bool | None = None,
        top_k: int | None = None,
        min_similarity: float | None = None,
    ) -> None:
        self.enabled = settings.enable_rag if enabled is None else enabled
        self.embeddings = embeddings or EmbeddingService()
        self.store = store or create_vector_store()
        self.pipeline = pipeline or DocumentPipeline(
            embeddings=self.embeddings,
            store=self.store,
            chunker=Chunker(),
        )
        self.retriever = retriever or Retriever(
            embeddings=self.embeddings,
            store=self.store,
            reranker=reranker or SimilarityReranker(),
            top_k=top_k,
            min_similarity=min_similarity,
        )
        self.context_builder = context_builder or RagContextBuilder()

    async def ingest(
        self,
        *,
        content: bytes,
        filename: str,
        document_id: str | None = None,
        source: str | None = None,
    ) -> IngestResult:
        if not self.enabled:
            return IngestResult(
                success=False,
                error="RAG is disabled",
                error_code=RagErrorCode.DISABLED,
            )
        return await self.pipeline.ingest(
            content=content,
            filename=filename,
            document_id=document_id,
            source=source,
        )

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        min_similarity: float | None = None,
        filters: dict[str, Any] | None = None,
    ) -> RetrievalResult:
        if not self.enabled:
            return RetrievalResult(
                success=False,
                query=query,
                error="RAG is disabled",
                error_code=RagErrorCode.DISABLED,
            )

        result = await self.retriever.retrieve(
            query,
            top_k=top_k,
            min_similarity=min_similarity,
            filters=filters,
        )
        if result.success:
            context_text = self.context_builder.build(result.hits)
            result = result.model_copy(
                update={
                    "context_text": context_text,
                    "context_chars": len(context_text),
                }
            )
            app_logger.info(
                f"RAG context_size={result.context_chars} "
                f"sources={len(result.hits)}"
            )
        return result

    def documents_for_context(
        self,
        result: RetrievalResult,
    ) -> list[str]:
        """Attributed chunk strings for platform ContextBuilder."""
        if not result.success or not result.hits:
            return []
        return self.context_builder.build_list(result.hits)

    async def list_documents(self) -> list[DocumentMeta]:
        catalog = self.pipeline.list_documents()
        if catalog:
            return catalog
        list_fn = getattr(self.store, "list_documents", None)
        if callable(list_fn):
            return await list_fn()
        return catalog

    async def get_document(self, document_id: str) -> DocumentMeta | None:
        found = self.pipeline.get_document(document_id)
        if found is not None:
            return found
        list_fn = getattr(self.store, "list_documents", None)
        if callable(list_fn):
            for meta in await list_fn():
                if meta.document_id == document_id:
                    return meta
        return None

    async def delete_document(self, document_id: str) -> bool:
        if not self.enabled:
            return False
        return await self.pipeline.delete(document_id)

    async def reindex(
        self,
        document_id: str | None = None,
    ) -> ReindexResult:
        if not self.enabled:
            return ReindexResult(
                success=False,
                error="RAG is disabled",
                error_code=RagErrorCode.DISABLED,
            )

        results = await self.pipeline.reindex(document_id)
        succeeded = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        return ReindexResult(
            success=len(failed) == 0 and len(results) > 0,
            reindexed=len(succeeded),
            failed=len(failed),
            document_ids=[
                r.document.document_id
                for r in succeeded
                if r.document is not None
            ],
            error=failed[0].error if failed else None,
            error_code=failed[0].error_code if failed else None,
        )

    @property
    def has_documents(self) -> bool:
        if len(self.pipeline.catalog) > 0:
            return True
        return int(getattr(self.store, "size", 0) or 0) > 0
