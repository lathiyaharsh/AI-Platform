from __future__ import annotations

from app.rag.models import RetrievalResult, SearchHit


class RagContextBuilder:
    """
    Build attributed context text from retrieved chunks.

    Output is consumed by the platform ContextBuilder as `documents`.
    Source attribution is always included for citations.
    """

    def build(
        self,
        hits: list[SearchHit] | RetrievalResult,
        *,
        max_chars: int | None = None,
    ) -> str:
        if isinstance(hits, RetrievalResult):
            items = hits.hits
        else:
            items = hits

        if not items:
            return ""

        blocks: list[str] = []
        total = 0

        for index, hit in enumerate(items, start=1):
            attribution = self._attribution(hit)
            block = (
                f"[Source {index}] {attribution}\n"
                f"{hit.text.strip()}\n"
                f"(similarity={hit.score:.4f})"
            )
            if max_chars is not None and total + len(block) + 2 > max_chars:
                break
            blocks.append(block)
            total += len(block) + 2

        return "\n\n".join(blocks)

    def build_list(
        self,
        hits: list[SearchHit] | RetrievalResult,
        *,
        max_chars: int | None = None,
    ) -> list[str]:
        """
        Return one attributed string per hit for ContextBuilder.documents.
        """
        text = self.build(hits, max_chars=max_chars)
        if not text:
            return []
        # Split back into per-source blocks while preserving attribution.
        parts = text.split("\n\n")
        return [part for part in parts if part.strip()]

    @staticmethod
    def _attribution(hit: SearchHit) -> str:
        parts: list[str] = []
        source = hit.source or hit.filename or hit.document_id
        if source:
            parts.append(f"source={source}")
        if hit.page is not None:
            parts.append(f"page={hit.page}")
        parts.append(f"document_id={hit.document_id}")
        parts.append(f"chunk_id={hit.chunk_id}")
        return " | ".join(parts)
