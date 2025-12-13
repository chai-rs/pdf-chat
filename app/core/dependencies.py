"""FastAPI dependency injection."""

from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from app.core.config import Settings, get_settings
from app.db.vector_store import VectorStore
from app.services.pdf_ingestion import PDFIngestionService
from app.services.web_search import WebSearchService


@lru_cache
def get_vector_store() -> VectorStore:
    """Get cached vector store instance."""
    settings = get_settings()
    return VectorStore(
        persist_directory=settings.chroma_persist_directory,
        embedding_model=settings.embedding_model,
    )


@lru_cache
def get_pdf_service() -> PDFIngestionService:
    """Get cached PDF ingestion service."""
    settings = get_settings()
    return PDFIngestionService(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )


@lru_cache
def get_web_search_service() -> WebSearchService:
    """Get cached web search service."""
    settings = get_settings()
    return WebSearchService(api_key=settings.tavily_api_key)


# Type aliases for dependency injection
SettingsDep = Annotated[Settings, Depends(get_settings)]
VectorStoreDep = Annotated[VectorStore, Depends(get_vector_store)]
PDFServiceDep = Annotated[PDFIngestionService, Depends(get_pdf_service)]
WebSearchDep = Annotated[WebSearchService, Depends(get_web_search_service)]
