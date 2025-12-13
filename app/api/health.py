"""Health check API routes."""

import logging

from fastapi import APIRouter, status

from app.core.dependencies import VectorStoreDep


logger = logging.getLogger(__name__)

router = APIRouter(tags=["Health"])


@router.get("/health", status_code=status.HTTP_200_OK)
async def health_check(vector_store: VectorStoreDep) -> dict:
    """Health check endpoint.

    Returns service status and vector store connection status.
    """
    try:
        _ = vector_store.vectorstore
        vector_status = "connected"
    except Exception as e:
        logger.warning(f"Vector store health check failed: {e}")
        vector_status = "disconnected"

    return {
        "status": "healthy",
        "vector_store": vector_status,
    }


@router.get("/", status_code=status.HTTP_200_OK)
async def root() -> dict:
    """Root endpoint with API information."""
    return {
        "message": "Chat with PDF API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }
