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

    # ==========================
    # Reflection
    # ==========================
    reflection_provider: str = "groq"
    reflection_model: str = "llama-3.3-70b-versatile"
    reflection_threshold: float = 0.7
    max_reflections: int = 1

    # ==========================
    # Retry
    # ==========================

    max_retries: int = 2
    retry_delay: float = 1.0

    # ==========================
    # Embeddings
    # ==========================

    embedding_provider: str = "huggingface"
    hf_embed_model: str = "BAAI/bge-small-en-v1.5"
    hf_embed_dim: int = 384
    semantic_cache_threshold: float = 0.90

    # ==========================
    # Memory
    # ==========================

    memory_window: int = 10
    long_term_memory_max_facts: int = 20

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
