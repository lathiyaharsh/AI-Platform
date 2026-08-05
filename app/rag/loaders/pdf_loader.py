from __future__ import annotations

from io import BytesIO

from pypdf import PdfReader

from app.rag.loaders.base import BaseDocumentLoader
from app.rag.loaders.text_loader import UnsupportedDocumentError
from app.rag.models import LoadedDocument, LoadedPage
from app.rag.types import DocumentType, RagErrorCode


class PdfLoader(BaseDocumentLoader):
    """Load PDF documents page-by-page via pypdf."""

    supported_types = frozenset({DocumentType.PDF})

    async def load(
        self,
        *,
        content: bytes,
        filename: str,
        document_id: str,
        source: str | None = None,
    ) -> LoadedDocument:
        if not content:
            raise UnsupportedDocumentError(
                f"PDF '{filename}' is empty",
                code=RagErrorCode.CORRUPTED_DOCUMENT,
            )

        try:
            reader = PdfReader(BytesIO(content))
        except Exception as exc:
            raise UnsupportedDocumentError(
                f"Corrupted or unreadable PDF '{filename}': {exc}",
                code=RagErrorCode.CORRUPTED_DOCUMENT,
            ) from exc

        if getattr(reader, "is_encrypted", False):
            try:
                reader.decrypt("")
            except Exception as exc:
                raise UnsupportedDocumentError(
                    f"Encrypted PDF '{filename}' cannot be opened: {exc}",
                    code=RagErrorCode.CORRUPTED_DOCUMENT,
                ) from exc

        pages: list[LoadedPage] = []
        try:
            for index, page in enumerate(reader.pages, start=1):
                try:
                    text = page.extract_text() or ""
                except Exception:
                    text = ""
                pages.append(LoadedPage(page=index, text=text))
        except Exception as exc:
            raise UnsupportedDocumentError(
                f"Failed extracting text from PDF '{filename}': {exc}",
                code=RagErrorCode.CORRUPTED_DOCUMENT,
            ) from exc

        if not any(p.text.strip() for p in pages):
            raise UnsupportedDocumentError(
                f"PDF '{filename}' contains no extractable text",
                code=RagErrorCode.CORRUPTED_DOCUMENT,
            )

        return LoadedDocument(
            document_id=document_id,
            filename=filename,
            source=self.resolve_source(filename, source),
            document_type=DocumentType.PDF,
            pages=pages,
            metadata={"page_count": len(pages)},
        )
