from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator


class DocumentUploadResponse(BaseModel):
    success: bool
    document_id: str | None = None
    filename: str | None = None
    chunk_count: int = 0
    upload_ms: float = 0.0
    embed_ms: float = 0.0
    store_ms: float = 0.0
    error: str | None = None
    error_code: str | None = None


class DocumentInfo(BaseModel):
    document_id: str
    filename: str
    source: str
    document_type: str
    chunk_count: int
    char_count: int
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentListResponse(BaseModel):
    documents: list[DocumentInfo]
    count: int


class DocumentDeleteResponse(BaseModel):
    success: bool
    document_id: str
    error: str | None = None


class ReindexRequest(BaseModel):
    document_id: str | None = None


class ReindexResponse(BaseModel):
    success: bool
    reindexed: int = 0
    failed: int = 0
    document_ids: list[str] = Field(default_factory=list)
    error: str | None = None
    error_code: str | None = None


class RagChatRequest(BaseModel):
    """
    RAG chat body.

    Leave top_k / min_similarity empty to use server settings
    (TOP_K=5, MIN_SIMILARITY=0.55). Do not set min_similarity=1.0 —
    that requires a perfect vector match and almost always returns
    zero hits (Swagger often pre-fills 1.0; those values are ignored).
    """

    message: str = Field(
        ...,
        min_length=1,
        examples=["what is the school name?"],
    )
    session_id: str | None = Field(
        default=None,
        examples=[None],
        description="Omit or null to start a new session",
    )
    user_id: str | None = Field(default=None, examples=[None])
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=50,
        examples=[5],
        description="Optional override; default comes from TOP_K settings",
    )
    min_similarity: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        examples=[0.55],
        description=(
            "Optional cosine threshold. Omit to use MIN_SIMILARITY from settings. "
            "Values >= 0.99 (including Swagger's default 1.0) fall back to settings."
        ),
    )

    @field_validator("session_id", "user_id", mode="before")
    @classmethod
    def _drop_swagger_placeholders(cls, value: Any) -> Any:
        if isinstance(value, str) and value.strip().lower() in {"", "string", "null"}:
            return None
        return value

    @field_validator("min_similarity", mode="before")
    @classmethod
    def _drop_impractical_threshold(cls, value: Any) -> Any:
        if value is None or value == "":
            return None
        try:
            score = float(value)
        except (TypeError, ValueError):
            return value
        # Perfect match is unrealistic; Swagger often pre-fills 1.
        if score >= 0.99:
            return None
        return score

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "message": "what is the school name?",
                    "top_k": 5,
                    "min_similarity": 0.55,
                }
            ]
        }
    }


class RagChatResponse(BaseModel):
    session_id: str
    response: str
    sources: list[dict[str, Any]] = Field(default_factory=list)
    retrieval_ms: float = 0.0
    context_chars: int = 0
