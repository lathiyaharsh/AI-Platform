from fastapi import FastAPI

from app.api.middleware.logging import LoggingMiddleware
from app.api.routes import chat, rag
from app.config.settings import settings

app = FastAPI(title=settings.app_name)

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
