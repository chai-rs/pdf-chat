"""Application configuration using Pydantic Settings."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API Keys
    openai_api_key: str = ""
    openai_api_base: str = "https://api.openai.com/v1"
    tavily_api_key: str = ""

    # ChromaDB settings
    chroma_persist_directory: str = "./data/chroma"

    # Model settings
    default_llm_model: str = "openai/gpt-5-mini"
    embedding_model: str = "openai/text-embedding-3-small"

    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # RAG settings
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k_results: int = 5


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()
