"""FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.chat import router as chat_router
from app.api.documents import router as documents_router
from app.api.health import router as health_router
from app.api.sessions import router as sessions_router
from app.core.config import settings


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info(f"Starting Chat with PDF API on {settings.api_host}:{settings.api_port}")
    yield
    logger.info("Shutting down Chat with PDF API")


app = FastAPI(
    title="Chat with PDF",
    description="RAG-powered chat with PDF documents using LangGraph",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# API Routes (v1)
# ============================================================================
# All routes follow RESTful conventions with /api/v1 prefix
#
# Health & Root:
#   GET  /                              - API info
#   GET  /health                        - Health check
#
# Chat:
#   POST /api/v1/chat                   - Chat with documents
#
# Sessions:
#   GET    /api/v1/sessions             - List all sessions
#   POST   /api/v1/sessions             - Create new session
#   GET    /api/v1/sessions/{id}        - Get session history
#   DELETE /api/v1/sessions/{id}        - Clear session memory
#   DELETE /api/v1/sessions/{id}/permanently - Delete session
#
# Documents:
#   POST   /api/v1/documents/upload     - Upload PDF
#   POST   /api/v1/documents/search     - Search documents
#   DELETE /api/v1/documents/collection - Delete all documents
# ============================================================================

# Root level routes (no prefix)
app.include_router(health_router)

# API v1 routes
app.include_router(chat_router, prefix="/api/v1/chat")
app.include_router(sessions_router, prefix="/api/v1/sessions")
app.include_router(documents_router, prefix="/api/v1/documents")
