"""FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.chat import router as chat_router
from app.api.documents import router as documents_router
from app.api.routes import router as api_router
from app.core.config import settings
from app.models.schemas import HealthResponse


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    logger.info(f"Starting Chat with PDF API on {settings.api_host}:{settings.api_port}")
    yield
    # Shutdown
    logger.info("Shutting down Chat with PDF API")


app = FastAPI(
    title="Chat with PDF",
    description="RAG-powered chat with PDF documents using LangGraph",
    version="0.1.0",
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

# Include routers
# New main API routes (POST /chat, POST /clear-memory, GET /health, GET /sessions/{session_id}/history)
app.include_router(api_router)
# Legacy routes for backward compatibility
app.include_router(chat_router, prefix="/api/v1")
app.include_router(documents_router, prefix="/api/v1")


@app.get("/", tags=["Root"])
async def root() -> dict:
    """Root endpoint."""
    return {
        "message": "Chat with PDF API",
        "docs": "/docs",
        "health": "/health",
    }
