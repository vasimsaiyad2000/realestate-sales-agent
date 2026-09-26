from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "realestate-sales-agent"
    environment: str = "development"

    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/realestate-sales-agent"

    openai_api_key: str | None = None
    openai_chat_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    whatsapp_app_secret: str | None = None
    whatsapp_verify_token: str = "replace-me"
    whatsapp_graph_version: str = "v21.0"

    google_maps_api_key: str | None = None

    embedding_provider: str = "ollama"
    embedding_dimensions: int = 768
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_embedding_model: str = "nomic-embed-text"
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    ollama_llm_model :str = "deepseek-r1:7b"
    nvidia_api_key: str | None = None
    nvidia_ca_bundle: str | None = None
    nemotron_embedding_model: str = "nvidia/llama-3.2-nv-embedqa-1b-v2"
    rag_sync_interval_seconds: int = 3600

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()