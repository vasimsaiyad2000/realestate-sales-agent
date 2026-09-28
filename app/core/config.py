from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Application
    app_name: str
    environment: str

    # Database
    database_url: str
    redis_url: str

    # WhatsApp
    whatsapp_app_secret: str | None
    whatsapp_verify_token: str
    whatsapp_graph_version: str

    # Google Maps
    google_maps_api_key: str | None

    # Embedding
    embedding_provider: str
    embedding_dimensions: int
    
    # Open Router and OpenAI
    open_router_base_url : str
    open_router_api_key : str
    openai_embedding_model: str

    # RAG
    rag_sync_interval_seconds: int

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()