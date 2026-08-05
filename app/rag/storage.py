from __future__ import annotations

from typing import Any

from app.observability.logger import app_logger
from app.rag.models import SearchHit, VectorRecord
from app.utils.similarity import cosine_similarity


class MemoryVectorStore:
    """
    In-memory vector store with cosine similarity search.

    Swappable via the VectorStore protocol for pgvector / Qdrant /
    Pinecone / Weaviate / Milvus / FAISS without changing RAGService.
    """

    def __init__(self) -> None:
        self._records: list[VectorRecord] = []

    async def add(
        self,
        *,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        if not (len(ids) == len(embeddings) == len(documents) == len(metadatas)):
            raise ValueError(
                "ids, embeddings, documents, and metadatas must be the same length"
            )

        for record_id, embedding, text, metadata in zip(
            ids,
            embeddings,
            documents,
            metadatas,
            strict=True,
        ):
            document_id = str(metadata.get("document_id", ""))
            chunk_id = str(metadata.get("chunk_id", record_id))
            self._records.append(
                VectorRecord(
                    id=record_id,
                    document_id=document_id,
                    chunk_id=chunk_id,
                    text=text,
                    embedding=list(embedding),
                    metadata=dict(metadata),
                )
            )

        app_logger.info(f"VectorStore ADD count={len(ids)} total={len(self._records)}")

    async def search(
        self,
        embedding: list[float],
        *,
        top_k: int = 5,
        min_similarity: float = 0.0,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchHit]:
        if top_k <= 0 or not self._records:
            return []

        scored: list[SearchHit] = []
        for record in self._records:
            if filters and not self._matches_filters(record.metadata, filters):
                continue
            try:
                score = float(cosine_similarity(embedding, record.embedding))
            except Exception:
                continue
            if score < min_similarity:
                continue
            scored.append(
                SearchHit(
                    document_id=record.document_id,
                    chunk_id=record.chunk_id,
                    text=record.text,
                    score=score,
                    metadata=dict(record.metadata),
                    page=record.metadata.get("page"),
                    source=str(record.metadata.get("source", "")),
                    filename=str(record.metadata.get("filename", "")),
                )
            )

        scored.sort(key=lambda hit: hit.score, reverse=True)
        return scored[:top_k]

    async def delete(self, document_id: str) -> int:
        before = len(self._records)
        self._records = [
            record for record in self._records if record.document_id != document_id
        ]
        removed = before - len(self._records)
        app_logger.info(
            f"VectorStore DELETE document_id={document_id} removed={removed}"
        )
        return removed

    async def clear(self) -> None:
        self._records.clear()
        app_logger.info("VectorStore CLEAR")

    @property
    def size(self) -> int:
        return len(self._records)

    @staticmethod
    def _matches_filters(
        metadata: dict[str, Any],
        filters: dict[str, Any],
    ) -> bool:
        for key, expected in filters.items():
            if metadata.get(key) != expected:
                return False
        return True
