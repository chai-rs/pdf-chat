"""API routes package."""

from app.api.chat import router as chat_router
from app.api.documents import router as documents_router
from app.api.health import router as health_router
from app.api.sessions import router as sessions_router

__all__ = [
    "chat_router",
    "documents_router",
    "health_router",
    "sessions_router",
]
