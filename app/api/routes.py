"""API routes for the Chat with PDF application.

This module provides the main API endpoints:
- POST /chat - Chat with documents
- POST /clear-memory - Clear session memory
- GET /health - Health check
- GET /sessions/{session_id}/history - Get conversation history
"""

import logging
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.agents.orchestrator import Orchestrator
from app.core.dependencies import VectorStoreDep, WebSearchDep
from app.db import memory as session_memory


logger = logging.getLogger(__name__)

router = APIRouter(tags=["API"])


# --- Request/Response Models ---


class ChatRequest(BaseModel):
    """Chat request model."""

    question: str = Field(..., description="User question to ask")
    session_id: str | None = Field(
        default=None, description="Session ID for conversation continuity"
    )


class SourceInfo(BaseModel):
    """Source information from retrieval."""

    content: str = Field(description="Content snippet from source")
    source_type: str = Field(description="Type of source (pdf/web)")
    metadata: dict = Field(default_factory=dict, description="Source metadata")


class ChatResponse(BaseModel):
    """Chat response model."""

    answer: str = Field(description="AI-generated answer")
    sources: list[SourceInfo] = Field(
        default_factory=list, description="Sources used for the answer"
    )
    agent_used: str = Field(description="Agent that handled the request")
    session_id: str = Field(description="Session ID for this conversation")
    needs_clarification: bool = Field(
        default=False, description="Whether clarification is needed"
    )
    clarification_question: str | None = Field(
        default=None, description="Clarification question if needed"
    )


class ClearMemoryRequest(BaseModel):
    """Request to clear session memory."""

    session_id: str = Field(..., description="Session ID to clear")


class ClearMemoryResponse(BaseModel):
    """Response from clearing memory."""

    success: bool = Field(description="Whether the operation succeeded")
    message: str = Field(description="Status message")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(description="Service status")
    vector_store: str = Field(description="Vector store connection status")


class MessageInfo(BaseModel):
    """Individual message in history."""

    role: str = Field(description="Message role (human/ai)")
    content: str = Field(description="Message content")


class SessionHistoryResponse(BaseModel):
    """Session history response."""

    session_id: str = Field(description="Session ID")
    message_count: int = Field(description="Number of messages")
    messages: list[MessageInfo] = Field(
        default_factory=list, description="Conversation messages"
    )


# --- Dependency Injection ---


def get_orchestrator() -> Orchestrator:
    """Get orchestrator instance."""
    return Orchestrator()


OrchestratorDep = Annotated[Orchestrator, Depends(get_orchestrator)]


# --- Endpoints ---


@router.post("/chat", response_model=ChatResponse, status_code=status.HTTP_200_OK)
async def chat(
    request: ChatRequest,
    orchestrator: OrchestratorDep,
    vector_store: VectorStoreDep,
) -> ChatResponse:
    """Chat with the PDF documents.

    This endpoint processes user questions using the orchestrator which:
    1. Checks if clarification is needed
    2. Routes to appropriate agent (PDF, Web, or Hybrid)
    3. Synthesizes an answer from retrieved content

    Auto-creates a session if session_id is not provided.
    """
    # Generate session ID if not provided
    session_id = request.session_id or str(uuid4())

    # Get chat history for context
    chat_history = session_memory.get_history(session_id)

    logger.info(
        f"Processing chat request - session: {session_id}, "
        f"history_length: {len(chat_history)}"
    )

    try:
        # Run the orchestrator
        result = await orchestrator.arun(
            question=request.question,
            session_id=session_id,
            chat_history=chat_history,
        )

        # Check if clarification is needed
        if result.get("needs_clarification", False):
            logger.info(f"Clarification needed for session {session_id}")
            return ChatResponse(
                answer=result.get("clarification_question", "Could you clarify?"),
                sources=[],
                agent_used="clarification",
                session_id=session_id,
                needs_clarification=True,
                clarification_question=result.get("clarification_question"),
            )

        # Extract answer and sources
        answer = result.get("final_answer", "I couldn't generate an answer.")
        route = result.get("route", "unknown")

        # Process sources from both PDF and web results
        sources: list[SourceInfo] = []

        # Add PDF sources
        for pdf_result in result.get("pdf_results", []):
            sources.append(
                SourceInfo(
                    content=pdf_result.get("content", "")[:300],
                    source_type="pdf",
                    metadata=pdf_result.get("metadata", {}),
                )
            )

        # Add web sources
        for web_result in result.get("web_results", []):
            sources.append(
                SourceInfo(
                    content=web_result.get("content", web_result.get("snippet", ""))[:300],
                    source_type="web",
                    metadata={
                        "title": web_result.get("title", ""),
                        "url": web_result.get("url", ""),
                    },
                )
            )

        # Save to conversation history
        session_memory.add_message(session_id, "human", request.question)
        session_memory.add_message(session_id, "ai", answer)

        logger.info(
            f"Chat completed - session: {session_id}, route: {route}, "
            f"sources: {len(sources)}"
        )

        return ChatResponse(
            answer=answer,
            sources=sources,
            agent_used=route,
            session_id=session_id,
            needs_clarification=False,
            clarification_question=None,
        )

    except Exception as e:
        logger.exception(f"Error processing chat request: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process chat request: {str(e)}",
        )


@router.post(
    "/clear-memory",
    response_model=ClearMemoryResponse,
    status_code=status.HTTP_200_OK,
)
async def clear_memory(request: ClearMemoryRequest) -> ClearMemoryResponse:
    """Clear conversation memory for a session.

    Removes all messages from the specified session's history.
    """
    logger.info(f"Clearing memory for session: {request.session_id}")

    try:
        success = session_memory.clear_memory(request.session_id)

        if success:
            return ClearMemoryResponse(
                success=True,
                message=f"Successfully cleared memory for session {request.session_id}",
            )
        else:
            return ClearMemoryResponse(
                success=False,
                message=f"Session {request.session_id} not found or already empty",
            )

    except Exception as e:
        logger.exception(f"Error clearing memory: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear memory: {str(e)}",
        )


@router.get("/health", response_model=HealthResponse, status_code=status.HTTP_200_OK)
async def health_check(vector_store: VectorStoreDep) -> HealthResponse:
    """Health check endpoint.

    Verifies the service is running and vector store is connected.
    """
    try:
        # Try to access the vector store to verify connection
        # This will raise an exception if not connected
        _ = vector_store.vectorstore
        vector_status = "connected"
    except Exception as e:
        logger.warning(f"Vector store health check failed: {e}")
        vector_status = "disconnected"

    return HealthResponse(
        status="healthy",
        vector_store=vector_status,
    )


@router.get(
    "/sessions/{session_id}/history",
    response_model=SessionHistoryResponse,
    status_code=status.HTTP_200_OK,
)
async def get_session_history(session_id: str) -> SessionHistoryResponse:
    """Get conversation history for a session.

    Useful for debugging and reviewing past conversations.
    """
    logger.info(f"Retrieving history for session: {session_id}")

    try:
        history = session_memory.get_history(session_id)

        messages = [
            MessageInfo(
                role="human" if hasattr(msg, "type") and msg.type == "human" else (
                    "ai" if hasattr(msg, "type") and msg.type == "ai" else
                    "human" if "HumanMessage" in type(msg).__name__ else "ai"
                ),
                content=msg.content,
            )
            for msg in history
        ]

        return SessionHistoryResponse(
            session_id=session_id,
            message_count=len(messages),
            messages=messages,
        )

    except Exception as e:
        logger.exception(f"Error retrieving session history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve session history: {str(e)}",
        )
