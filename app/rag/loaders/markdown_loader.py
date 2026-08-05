from __future__ import annotations

from app.rag.loaders.base import BaseDocumentLoader
from app.rag.loaders.text_loader import UnsupportedDocumentError
from app.rag.models import LoadedDocument, LoadedPage
from app.rag.types import DocumentType


class MarkdownLoader(BaseDocumentLoader):
    """Load Markdown (.md / .markdown) documents as text."""

    supported_types = frozenset({DocumentType.MD})

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
                    f"Unable to decode markdown file '{filename}': {exc}",
                ) from exc

        return LoadedDocument(
            document_id=document_id,
            filename=filename,
            source=self.resolve_source(filename, source),
            document_type=DocumentType.MD,
            pages=[LoadedPage(page=1, text=text)],
            metadata={"format": "markdown"},
        )
