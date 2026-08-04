from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from .env"""

    # ==========================
    # App
    # ==========================
    app_name: str = "Production AI Platform"
    app_env: str = "development"

    host: str = "0.0.0.0"
    port: int = 8000

    log_level: str = "INFO"

    # ==========================
    # Providers
    # ==========================
    groq_api_key: str = ""
    gemini_api_key: str = ""
    huggingface_api_key: str = ""

    # ==========================
    # Database
    # ==========================
    supabase_db_url: str = ""

    # ==========================
    # AI
    # ==========================
    default_provider: str = "groq"
    default_model: str = "llama-3.3-70b-versatile"

    # ==========================
    # Features
    # ==========================
    cache_enabled: bool = True
    semantic_cache: bool = True
    reflection_enabled: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """
    Returns a cached Settings instance.
    """
    return Settings()


settings = get_settings()