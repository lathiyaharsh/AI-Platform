from fastapi import FastAPI

from app.config.settings import settings

app = FastAPI(title=settings.app_name)


@app.get("/")
async def root():
    return {
        "message": "Production AI Platform",
        "environment": settings.app_env,
        "provider": settings.default_provider,
        "model": settings.default_model,
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
    }   