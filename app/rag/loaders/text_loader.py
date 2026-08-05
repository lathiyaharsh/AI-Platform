from __future__ import annotations

from app.rag.loaders.base import BaseDocumentLoader
from app.rag.models import LoadedDocument, LoadedPage
from app.rag.types import DocumentType, RagErrorCode


class UnsupportedDocumentError(Exception):
    """Raised when content cannot be decoded as text."""

    def __init__(self, message: str, code: RagErrorCode = RagErrorCode.CORRUPTED_DOCUMENT):
        super().__init__(message)
        self.code = code


class TextLoader(BaseDocumentLoader):
    """Load plain-text (.txt / .text) documents."""

    supported_types = frozenset({DocumentType.TXT})

    async def load(
        self,
        *,
        content: bytes,
        filename: str,
        document_id: str,
        source: str | None = None,
    ) -> LoadedDocument:
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = content.decode("latin-1")
            except Exception as exc:
                raise UnsupportedDocumentError(
                    f"Unable to decode text file '{filename}': {exc}",
                ) from exc

        return LoadedDocument(
            document_id=document_id,
            filename=filename,
            source=self.resolve_source(filename, source),
            document_type=DocumentType.TXT,
            pages=[LoadedPage(page=1, text=text)],
            metadata={"encoding": "utf-8"},
        )
