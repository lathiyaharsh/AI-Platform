from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.dependencies.gateway import get_rag_service
from app.api.middleware.logging import LoggingMiddleware
from app.api.routes import chat, rag
from app.config.settings import settings
from app.db.postgres import close_pool
from app.observability.logger import app_logger


@asynccontextmanager
async def lifespan(_app: FastAPI):
    rag = get_rag_service()
    ensure = getattr(rag.store, "_ensure_ready", None)
    if callable(ensure):
        try:
            await ensure()
            app_logger.info(
                f"RAG vector store ready backend={settings.vector_store} "
                f"size={getattr(rag.store, 'size', 0)}"
            )
        except Exception as exc:
            app_logger.error(f"RAG vector store warmup failed: {exc}")
    yield
    await close_pool()


app = FastAPI(title=settings.app_name, lifespan=lifespan)

app.add_middleware(LoggingMiddleware)

app.include_router(chat.router)
app.include_router(rag.router)


@app.get("/")
async def root():
    return {
        "message": "Production AI Platform",
        "environment": settings.app_env,
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}
