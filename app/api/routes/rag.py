from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.api.dependencies.gateway import get_gateway, get_rag_service
from app.api.schemas.chat import ChatResponse
from app.api.schemas.rag import (
    DocumentDeleteResponse,
    DocumentInfo,
    DocumentListResponse,
    DocumentUploadResponse,
    RagChatRequest,
    ReindexRequest,
    ReindexResponse,
)
from app.gateway.gateway import Gateway
from app.rag.service import RAGService
from app.rag.types import RagErrorCode

router = APIRouter(tags=["rag"])


def _to_document_info(meta) -> DocumentInfo:
    return DocumentInfo(
        document_id=meta.document_id,
        filename=meta.filename,
        source=meta.source,
        document_type=meta.document_type.value,
        chunk_count=meta.chunk_count,
        char_count=meta.char_count,
        created_at=meta.created_at,
        updated_at=meta.updated_at,
        metadata=meta.metadata,
    )


@router.post(
    "/documents/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_document(
    file: UploadFile = File(...),
    rag: RAGService = Depends(get_rag_service),
):
    """Ingest an uploaded document into the RAG pipeline."""
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename is required",
        )

    content = await file.read()
    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty",
        )

    result = await rag.ingest(
        content=content,
        filename=file.filename,
        source=file.filename,
    )

    if not result.success:
        status_code = status.HTTP_400_BAD_REQUEST
        if result.error_code == RagErrorCode.DISABLED:
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        elif result.error_code == RagErrorCode.EMBEDDING_FAILURE:
            status_code = status.HTTP_502_BAD_GATEWAY
        elif result.error_code == RagErrorCode.STORAGE_FAILURE:
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        raise HTTPException(
            status_code=status_code,
            detail={
                "error": result.error,
                "error_code": (
                    result.error_code.value if result.error_code else None
                ),
            },
        )

    document = result.document
    return DocumentUploadResponse(
        success=True,
        document_id=document.document_id if document else None,
        filename=document.filename if document else file.filename,
        chunk_count=result.chunks_created,
        upload_ms=result.upload_ms,
        embed_ms=result.embed_ms,
        store_ms=result.store_ms,
    )


@router.post("/documents/reindex", response_model=ReindexResponse)
async def reindex_documents(
    request: ReindexRequest | None = None,
    rag: RAGService = Depends(get_rag_service),
):
    """Re-chunk and re-embed one document or the full catalog."""
    body = request or ReindexRequest()
    result = await rag.reindex(body.document_id)

    if not result.success and result.error_code == RagErrorCode.DISABLED:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=result.error,
        )

    return ReindexResponse(
        success=result.success,
        reindexed=result.reindexed,
        failed=result.failed,
        document_ids=result.document_ids,
        error=result.error,
        error_code=result.error_code.value if result.error_code else None,
    )


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    rag: RAGService = Depends(get_rag_service),
):
    documents = await rag.list_documents()
    items = [_to_document_info(meta) for meta in documents]
    return DocumentListResponse(documents=items, count=len(items))


@router.delete(
    "/documents/{document_id}",
    response_model=DocumentDeleteResponse,
)
async def delete_document(
    document_id: str,
    rag: RAGService = Depends(get_rag_service),
):
    deleted = await rag.delete_document(document_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found",
        )
    return DocumentDeleteResponse(success=True, document_id=document_id)


@router.post("/chat/rag", response_model=ChatResponse)
async def chat_rag(
    request: RagChatRequest,
    gateway: Gateway = Depends(get_gateway),
):
    """
    Chat with forced RAG retrieval.

    Uses the same Gateway path as /chat with force_rag=True.
    """
    session_id = request.session_id or str(uuid4())

    answer = await gateway.generate(
        prompt=request.message,
        session_id=session_id,
        user_id=request.user_id,
        force_rag=True,
        top_k=request.top_k,
        min_similarity=request.min_similarity,
    )

    return ChatResponse(
        session_id=session_id,
        response=answer,
    )
