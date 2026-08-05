from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import PurePosixPath

from app.rag.constants import EXTENSION_MAP
from app.rag.models import LoadedDocument
from app.rag.types import DocumentType


class BaseDocumentLoader(ABC):
    """
    Abstract document loader.

    Subclasses declare supported DocumentType values and implement load().
    New formats (docx, html, csv) subclass this and register in the pipeline.
    """

    supported_types: frozenset[DocumentType]

    @abstractmethod
    async def load(
        self,
        *,
        content: bytes,
        filename: str,
        document_id: str,
        source: str | None = None,
    ) -> LoadedDocument:
        raise NotImplementedError

    @staticmethod
    def detect_type(filename: str) -> DocumentType:
        suffix = PurePosixPath(filename.lower()).suffix
        mapped = EXTENSION_MAP.get(suffix)
        if mapped is None:
            return DocumentType.UNKNOWN
        return DocumentType(mapped)

    @staticmethod
    def resolve_source(filename: str, source: str | None) -> str:
        return source or filename
