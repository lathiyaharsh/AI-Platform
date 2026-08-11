from __future__ import annotations

import json
from typing import Any

from app.config.settings import settings
from app.db.postgres import get_pool
from app.observability.logger import app_logger
from app.rag.models import DocumentMeta, SearchHit
from app.rag.types import DocumentType


def _as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        parsed = json.loads(value) if value else {}
        return dict(parsed) if isinstance(parsed, dict) else {}
    if isinstance(value, (bytes, bytearray)):
        parsed = json.loads(value.decode("utf-8"))
        return dict(parsed) if isinstance(parsed, dict) else {}
    try:
        return dict(value)
    except Exception:
        return {}


class PgVectorStore:
    """
    Postgres + pgvector store (Supabase-compatible).

    Implements the VectorStore protocol used by DocumentPipeline / Retriever.
    Collections isolate tenants/environments via SUPABASE_COLLECTION.
    """

    def __init__(
        self,
        *,
        collection: str | None = None,
        dimension: int | None = None,
        table: str = "rag_chunks",
    ) -> None:
        self.collection = collection or settings.supabase_collection
        self.dimension = dimension if dimension is not None else settings.hf_embed_dim
        self.table = table
        self._ready = False
        self._size = 0

    @property
    def size(self) -> int:
        return self._size

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
        await self._ensure_ready()
        if not ids:
            return

        pool = await get_pool()
        rows: list[tuple[Any, ...]] = []
        for record_id, embedding, text, metadata in zip(
            ids,
            embeddings,
            documents,
            metadatas,
            strict=True,
        ):
            if len(embedding) != self.dimension:
                raise ValueError(
                    f"Embedding dim {len(embedding)} != configured {self.dimension}"
                )
            document_id = str(metadata.get("document_id", ""))
            chunk_id = str(metadata.get("chunk_id", record_id))
            rows.append(
                (
                    record_id,
                    self.collection,
                    document_id,
                    chunk_id,
                    text,
                    embedding,
                    json.dumps(metadata),
                )
            )

        async with pool.acquire() as conn:
            await conn.executemany(
                f"""
                INSERT INTO {self.table} (
                    id, collection, document_id, chunk_id, content, embedding, metadata
                )
                VALUES ($1, $2, $3, $4, $5, $6::vector, $7::jsonb)
                ON CONFLICT (id) DO UPDATE SET
                    collection = EXCLUDED.collection,
                    document_id = EXCLUDED.document_id,
                    chunk_id = EXCLUDED.chunk_id,
                    content = EXCLUDED.content,
                    embedding = EXCLUDED.embedding,
                    metadata = EXCLUDED.metadata
                """,
                rows,
            )
            self._size = int(
                await conn.fetchval(
                    f"SELECT COUNT(*) FROM {self.table} WHERE collection = $1",
                    self.collection,
                )
                or 0
            )

        app_logger.info(
            f"PgVectorStore ADD collection={self.collection} "
            f"count={len(rows)} total={self._size}"
        )

    async def search(
        self,
        embedding: list[float],
        *,
        top_k: int = 5,
        min_similarity: float = 0.0,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchHit]:
        if top_k <= 0:
            return []
        await self._ensure_ready()
        if len(embedding) != self.dimension:
            raise ValueError(
                f"Query embedding dim {len(embedding)} != configured {self.dimension}"
            )

        where = ["collection = $2", "1 - (embedding <=> $1::vector) >= $3"]
        args: list[Any] = [embedding, self.collection, min_similarity]
        arg_i = 4

        if filters:
            for key, expected in filters.items():
                where.append(f"metadata->>${arg_i} = ${arg_i + 1}")
                args.extend([str(key), str(expected)])
                arg_i += 2

        args.append(top_k)
        limit_placeholder = f"${arg_i}"
        sql = f"""
            SELECT
                document_id,
                chunk_id,
                content,
                metadata,
                1 - (embedding <=> $1::vector) AS score
            FROM {self.table}
            WHERE {" AND ".join(where)}
            ORDER BY embedding <=> $1::vector
            LIMIT {limit_placeholder}
        """

        pool = await get_pool()
        async with pool.acquire() as conn:
            records = await conn.fetch(sql, *args)

        hits: list[SearchHit] = []
        for row in records:
            metadata = _as_dict(row["metadata"])
            hits.append(
                SearchHit(
                    document_id=str(row["document_id"]),
                    chunk_id=str(row["chunk_id"]),
                    text=str(row["content"]),
                    score=float(row["score"]),
                    metadata=metadata,
                    page=metadata.get("page"),
                    source=str(metadata.get("source", "")),
                    filename=str(metadata.get("filename", "")),
                )
            )
        return hits

    async def delete(self, document_id: str) -> int:
        await self._ensure_ready()
        pool = await get_pool()
        async with pool.acquire() as conn:
            result = await conn.execute(
                f"""
                DELETE FROM {self.table}
                WHERE collection = $1 AND document_id = $2
                """,
                self.collection,
                document_id,
            )
        removed = int(result.split()[-1]) if result else 0
        self._size = max(0, self._size - removed)
        app_logger.info(
            f"PgVectorStore DELETE collection={self.collection} "
            f"document_id={document_id} removed={removed}"
        )
        return removed

    async def clear(self) -> None:
        await self._ensure_ready()
        pool = await get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                f"DELETE FROM {self.table} WHERE collection = $1",
                self.collection,
            )
        self._size = 0
        app_logger.info(f"PgVectorStore CLEAR collection={self.collection}")

    async def list_documents(self) -> list[DocumentMeta]:
        """Aggregate catalog entries from stored chunk metadata."""
        await self._ensure_ready()
        pool = await get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT
                    document_id,
                    COUNT(*)::int AS chunk_count,
                    COALESCE(SUM(char_length(content)), 0)::int AS char_count,
                    (array_agg(metadata ORDER BY created_at ASC))[1] AS meta,
                    MIN(created_at) AS created_at,
                    MAX(created_at) AS updated_at
                FROM {self.table}
                WHERE collection = $1
                GROUP BY document_id
                ORDER BY MIN(created_at) ASC
                """,
                self.collection,
            )

        documents: list[DocumentMeta] = []
        for row in rows:
            meta = _as_dict(row["meta"])
            doc_type_raw = str(meta.get("document_type", "unknown"))
            try:
                doc_type = DocumentType(doc_type_raw)
            except ValueError:
                doc_type = DocumentType.UNKNOWN
            documents.append(
                DocumentMeta(
                    document_id=str(row["document_id"]),
                    filename=str(meta.get("filename", row["document_id"])),
                    source=str(meta.get("source", "")),
                    document_type=doc_type,
                    chunk_count=int(row["chunk_count"]),
                    char_count=int(row["char_count"]),
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                    metadata=meta,
                )
            )
        return documents

    async def _ensure_ready(self) -> None:
        if self._ready:
            return
        if not settings.supabase_db_url:
            raise RuntimeError(
                "VECTOR_STORE=pgvector requires SUPABASE_DB_URL to be set"
            )

        pool = await get_pool()
        async with pool.acquire() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            await conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.table} (
                    id TEXT PRIMARY KEY,
                    collection TEXT NOT NULL,
                    document_id TEXT NOT NULL,
                    chunk_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    embedding vector({self.dimension}) NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            await conn.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {self.table}_collection_idx
                ON {self.table} (collection)
                """
            )
            await conn.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {self.table}_document_id_idx
                ON {self.table} (document_id)
                """
            )
            # HNSW may already exist with a different opclass; ignore conflicts.
            try:
                await conn.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS {self.table}_embedding_hnsw_idx
                    ON {self.table}
                    USING hnsw (embedding vector_cosine_ops)
                    """
                )
            except Exception as exc:
                app_logger.warning(
                    f"PgVectorStore HNSW index skipped/unavailable: {exc}"
                )

            count = await conn.fetchval(
                f"SELECT COUNT(*) FROM {self.table} WHERE collection = $1",
                self.collection,
            )
            self._size = int(count or 0)

        self._ready = True
        app_logger.info(
            f"PgVectorStore ready table={self.table} "
            f"collection={self.collection} dim={self.dimension} size={self._size}"
        )
