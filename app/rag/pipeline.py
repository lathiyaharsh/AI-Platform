from __future__ import annotations

import time
from pathlib import PurePosixPath
from uuid import uuid4

from app.embeddings.service import EmbeddingService
from app.observability.logger import app_logger
from app.rag.chunking import Chunker, clean_text
from app.rag.constants import SUPPORTED_UPLOAD_EXTENSIONS
from app.rag.loaders import (
    BaseDocumentLoader,
    MarkdownLoader,
    PdfLoader,
    TextLoader,
    UnsupportedDocumentError,
)
from app.rag.models import (
    DocumentMeta,
    IngestResult,
    LoadedDocument,
)
from app.rag.storage import MemoryVectorStore
from app.rag.types import DocumentType, RagErrorCode, VectorStore


class DocumentPipeline:
    """
    Document intelligence pipeline.

    Upload → Loader → Clean → Chunk → Embed → Store Vector

    Designed so future loaders (docx, html, csv) register without
    changing RAGService.
    """

    def __init__(
        self,
        *,
        embeddings: EmbeddingService | None = None,
        store: VectorStore | None = None,
        chunker: Chunker | None = None,
        loaders: list[BaseDocumentLoader] | None = None,
    ) -> None:
        self.embeddings = embeddings or EmbeddingService()
        self.store = store or MemoryVectorStore()
        self.chunker = chunker or Chunker()
        seed = loaders if loaders is not None else [
            TextLoader(),
            MarkdownLoader(),
            PdfLoader(),
        ]
        self._loaders: dict[DocumentType, BaseDocumentLoader] = {}
        for loader in seed:
            for doc_type in loader.supported_types:
                self._loaders[doc_type] = loader

        # Raw content retained for reindex (in-memory catalog backing).
        self._raw_content: dict[str, bytes] = {}
        self._catalog: dict[str, DocumentMeta] = {}

    @property
    def catalog(self) -> dict[str, DocumentMeta]:
        return self._catalog

    def register_loader(self, loader: BaseDocumentLoader) -> None:
        for doc_type in loader.supported_types:
            self._loaders[doc_type] = loader

    async def ingest(
        self,
        *,
        content: bytes,
        filename: str,
        document_id: str | None = None,
        source: str | None = None,
        replace: bool = True,
    ) -> IngestResult:
        started = time.perf_counter()
        document_id = document_id or str(uuid4())

        try:
            doc_type = self._resolve_type(filename)
            loader = self._loaders.get(doc_type)
            if loader is None:
                return IngestResult(
                    success=False,
                    error=(
                        f"Unsupported document type for '{filename}'. "
                        f"Supported: {sorted(SUPPORTED_UPLOAD_EXTENSIONS)}"
                    ),
                    error_code=RagErrorCode.UNSUPPORTED_DOCUMENT,
                    upload_ms=(time.perf_counter() - started) * 1000,
                )

            loaded = await loader.load(
                content=content,
                filename=filename,
                document_id=document_id,
                source=source,
            )
            loaded = self._clean_document(loaded)

            chunks = self.chunker.chunk_document(loaded)
            if not chunks:
                return IngestResult(
                    success=False,
                    error=f"No chunks produced for '{filename}'",
                    error_code=RagErrorCode.CORRUPTED_DOCUMENT,
                    upload_ms=(time.perf_counter() - started) * 1000,
                )

            if replace:
                await self.store.delete(document_id)

            embed_started = time.perf_counter()
            try:
                embeddings = await self.embeddings.embed_many(
                    [chunk.text for chunk in chunks]
                )
            except Exception as exc:
                app_logger.error(f"RAG embedding failure: {exc}")
                return IngestResult(
                    success=False,
                    error=str(exc),
                    error_code=RagErrorCode.EMBEDDING_FAILURE,
                    upload_ms=(time.perf_counter() - started) * 1000,
                )
            embed_ms = (time.perf_counter() - embed_started) * 1000

            store_started = time.perf_counter()
            try:
                await self.store.add(
                    ids=[chunk.chunk_id for chunk in chunks],
                    embeddings=embeddings,
                    documents=[chunk.text for chunk in chunks],
                    metadatas=[chunk.metadata for chunk in chunks],
                )
            except Exception as exc:
                app_logger.error(f"RAG vector storage failure: {exc}")
                return IngestResult(
                    success=False,
                    error=str(exc),
                    error_code=RagErrorCode.STORAGE_FAILURE,
                    upload_ms=(time.perf_counter() - started) * 1000,
                    embed_ms=embed_ms,
                )
            store_ms = (time.perf_counter() - store_started) * 1000

            meta = DocumentMeta(
                document_id=document_id,
                filename=filename,
                source=loaded.source,
                document_type=loaded.document_type,
                chunk_count=len(chunks),
                char_count=sum(len(c.text) for c in chunks),
                metadata=dict(loaded.metadata),
            )
            self._catalog[document_id] = meta
            self._raw_content[document_id] = content

            upload_ms = (time.perf_counter() - started) * 1000
            app_logger.info(
                f"RAG ingest document_id={document_id} "
                f"chunks={len(chunks)} upload_ms={upload_ms:.1f} "
                f"embed_ms={embed_ms:.1f} store_ms={store_ms:.1f}"
            )

            return IngestResult(
                success=True,
                document=meta,
                chunks_created=len(chunks),
                upload_ms=upload_ms,
                embed_ms=embed_ms,
                store_ms=store_ms,
            )

        except UnsupportedDocumentError as exc:
            app_logger.error(f"RAG unsupported/corrupted document: {exc}")
            return IngestResult(
                success=False,
                error=str(exc),
                error_code=getattr(exc, "code", RagErrorCode.CORRUPTED_DOCUMENT),
                upload_ms=(time.perf_counter() - started) * 1000,
            )
        except Exception as exc:
            app_logger.error(f"RAG ingest failure: {exc}")
            return IngestResult(
                success=False,
                error=str(exc),
                error_code=RagErrorCode.UNKNOWN,
                upload_ms=(time.perf_counter() - started) * 1000,
            )

    async def delete(self, document_id: str) -> bool:
        removed = await self.store.delete(document_id)
        in_catalog = document_id in self._catalog
        self._catalog.pop(document_id, None)
        self._raw_content.pop(document_id, None)
        return removed > 0 or in_catalog

    async def reindex(
        self,
        document_id: str | None = None,
    ) -> list[IngestResult]:
        targets: list[str]
        if document_id:
            targets = [document_id]
        else:
            targets = list(self._catalog.keys())

        results: list[IngestResult] = []
        for doc_id in targets:
            meta = self._catalog.get(doc_id)
            raw = self._raw_content.get(doc_id)
            if meta is None or raw is None:
                results.append(
                    IngestResult(
                        success=False,
                        error=f"Document '{doc_id}' not found for reindex",
                        error_code=RagErrorCode.DOCUMENT_NOT_FOUND,
                    )
                )
                continue
            results.append(
                await self.ingest(
                    content=raw,
                    filename=meta.filename,
                    document_id=doc_id,
                    source=meta.source,
                    replace=True,
                )
            )
        return results

    def list_documents(self) -> list[DocumentMeta]:
        return list(self._catalog.values())

    def get_document(self, document_id: str) -> DocumentMeta | None:
        return self._catalog.get(document_id)

    def _resolve_type(self, filename: str) -> DocumentType:
        detected = BaseDocumentLoader.detect_type(filename)
        if detected == DocumentType.UNKNOWN:
            return DocumentType.UNKNOWN
        suffix = PurePosixPath(filename.lower()).suffix
        if suffix not in SUPPORTED_UPLOAD_EXTENSIONS:
            return DocumentType.UNKNOWN
        return detected

    def _clean_document(self, document: LoadedDocument) -> LoadedDocument:
        cleaned_pages = []
        for page in document.pages:
            cleaned_pages.append(
                page.model_copy(update={"text": clean_text(page.text)})
            )
        return document.model_copy(update={"pages": cleaned_pages})
