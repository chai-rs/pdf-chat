"""Chat API routes."""

import logging
from uuid import uuid4

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.agents.orchestrator import Orchestrator
from app.core.dependencies import VectorStoreDep
from app.db import memory as session_memory


logger = logging.getLogger(__name__)

router = APIRouter(tags=["Chat"])


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
    needs_clarification: bool = Field(default=False, description="Whether clarification is needed")
    clarification_question: str | None = Field(
        default=None, description="Clarification question if needed"
    )


# --- Dependency ---


def get_orchestrator() -> Orchestrator:
    """Get orchestrator instance."""
    return Orchestrator()


# --- Endpoints ---


@router.post("", response_model=ChatResponse, status_code=status.HTTP_200_OK)
async def chat(
    request: ChatRequest,
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
        f"Processing chat request - session: {session_id}, history_length: {len(chat_history)}"
    )

    try:
        # Create and run orchestrator
        orchestrator = get_orchestrator()
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
            f"Chat completed - session: {session_id}, route: {route}, sources: {len(sources)}"
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
