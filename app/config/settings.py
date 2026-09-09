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
    supabase_collection: str = "ai_chat_docs"

    # ==========================
    # AI
    # ==========================
    default_provider: str = "groq"
    # Groq retired llama-3.3-70b-versatile (2026-08-16); use gpt-oss-120b.
    default_model: str = "openai/gpt-oss-120b"

    # ==========================
    # Features
    # ==========================
    cache_enabled: bool = True
    semantic_cache: bool = True
    reflection_enabled: bool = True
    tools_enabled: bool = True
    enable_rag: bool = True

    # ==========================
    # Reflection
    # ==========================
    reflection_provider: str = "groq"
    reflection_model: str = "openai/gpt-oss-120b"
    reflection_threshold: float = 0.7
    max_reflections: int = 1

    # ==========================
    # Tools
    # ==========================
    tools_provider: str = "groq"
    tools_model: str = "openai/gpt-oss-120b"
    tool_timeout_seconds: float = 15.0
    max_tool_iterations: int = 3

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

    # ==========================
    # RAG
    # ==========================

    top_k: int = 5
    # BGE cosine scores for related text often land ~0.55–0.75;
    # 0.75 filtered real retrieval hits (e.g. school-name queries ~0.69).
    min_similarity: float = 0.55
    chunk_size: int = 512
    chunk_overlap: int = 64
    chunk_strategy: str = "sentence"
    vector_store: str = "pgvector"

    # ==========================
    # Observability
    # ==========================

    observability_enabled: bool = True
    estimate_cost: bool = True
    log_request_body: bool = False
    log_response_body: bool = False
    enable_json_logging: bool = True

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
