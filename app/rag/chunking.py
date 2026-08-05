from __future__ import annotations

import re
from uuid import uuid4

from app.config.settings import settings
from app.rag.models import Chunk, LoadedDocument
from app.rag.types import ChunkStrategy

_SENTENCE_SPLIT_RE = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])|(?<=\n)\s*",
)
_WHITESPACE_RE = re.compile(r"[ \t]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    """Normalize whitespace and strip control characters."""
    cleaned = text.replace("\x00", "")
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = _WHITESPACE_RE.sub(" ", cleaned)
    cleaned = _MULTI_NEWLINE_RE.sub("\n\n", cleaned)
    return cleaned.strip()


class Chunker:
    """
    Configurable document chunker.

    Strategies
    - FIXED: character windows with overlap
    - SENTENCE: pack sentences into ~chunk_size windows with overlap

    Chunk metadata always includes document_id, chunk_id, page, source.
    """

    def __init__(
        self,
        *,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        strategy: ChunkStrategy | None = None,
    ) -> None:
        self.chunk_size = chunk_size if chunk_size is not None else settings.chunk_size
        self.chunk_overlap = (
            chunk_overlap if chunk_overlap is not None else settings.chunk_overlap
        )
        self.strategy = strategy or ChunkStrategy(settings.chunk_strategy)

        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be >= 0")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be < chunk_size")

    def chunk_document(self, document: LoadedDocument) -> list[Chunk]:
        chunks: list[Chunk] = []
        index = 0

        for page in document.pages:
            text = clean_text(page.text)
            if not text:
                continue

            pieces = self._split(text)
            for piece in pieces:
                chunk_id = f"{document.document_id}:{index}:{uuid4().hex[:8]}"
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        document_id=document.document_id,
                        text=piece,
                        index=index,
                        page=page.page,
                        source=document.source,
                        filename=document.filename,
                        document_type=document.document_type,
                        strategy=self.strategy,
                        metadata={
                            "document_id": document.document_id,
                            "chunk_id": chunk_id,
                            "page": page.page,
                            "source": document.source,
                            "filename": document.filename,
                            "document_type": document.document_type.value,
                            "index": index,
                        },
                    )
                )
                index += 1

        return chunks

    def _split(self, text: str) -> list[str]:
        if self.strategy == ChunkStrategy.FIXED:
            return self._fixed_chunks(text)
        return self._sentence_chunks(text)

    def _fixed_chunks(self, text: str) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text]

        chunks: list[str] = []
        start = 0
        length = len(text)
        step = self.chunk_size - self.chunk_overlap

        while start < length:
            end = min(start + self.chunk_size, length)
            piece = text[start:end].strip()
            if piece:
                chunks.append(piece)
            if end >= length:
                break
            start += step

        return chunks

    def _sentence_chunks(self, text: str) -> list[str]:
        sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
        if not sentences:
            return self._fixed_chunks(text)

        chunks: list[str] = []
        current: list[str] = []
        current_len = 0

        for sentence in sentences:
            sentence_len = len(sentence)
            # Oversized single sentence → fall back to fixed windows.
            if sentence_len > self.chunk_size:
                if current:
                    chunks.append(" ".join(current))
                    current, current_len = [], 0
                chunks.extend(self._fixed_chunks(sentence))
                continue

            extra = sentence_len if not current else sentence_len + 1
            if current and current_len + extra > self.chunk_size:
                chunks.append(" ".join(current))
                overlap_text = self._overlap_tail(" ".join(current))
                if overlap_text:
                    current = [overlap_text, sentence]
                    current_len = len(overlap_text) + 1 + sentence_len
                else:
                    current = [sentence]
                    current_len = sentence_len
            else:
                current.append(sentence)
                current_len += extra

        if current:
            chunks.append(" ".join(current))

        return chunks

    def _overlap_tail(self, text: str) -> str:
        if self.chunk_overlap <= 0 or not text:
            return ""
        if len(text) <= self.chunk_overlap:
            return text
        return text[-self.chunk_overlap :].strip()
