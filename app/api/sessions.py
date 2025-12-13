"""Session management API routes."""

import logging
from uuid import uuid4

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.db import memory as session_memory


logger = logging.getLogger(__name__)

router = APIRouter(tags=["Sessions"])


# --- Request/Response Models ---


class ClearMemoryRequest(BaseModel):
    """Request to clear session memory."""

    session_id: str = Field(..., description="Session ID to clear")


class ClearMemoryResponse(BaseModel):
    """Response from clearing memory."""

    success: bool = Field(description="Whether the operation succeeded")
    message: str = Field(description="Status message")


class MessageInfo(BaseModel):
    """Individual message in history."""

    role: str = Field(description="Message role (human/ai)")
    content: str = Field(description="Message content")


class SessionHistoryResponse(BaseModel):
    """Session history response."""

    session_id: str = Field(description="Session ID")
    message_count: int = Field(description="Number of messages")
    messages: list[MessageInfo] = Field(default_factory=list, description="Conversation messages")


class SessionListResponse(BaseModel):
    """List of active sessions."""

    sessions: list[str] = Field(description="List of session IDs")
    count: int = Field(description="Number of active sessions")


# --- Endpoints ---


@router.get("", response_model=SessionListResponse, status_code=status.HTTP_200_OK)
async def list_sessions() -> SessionListResponse:
    """List all active sessions."""
    session_ids = session_memory.get_session_ids()
    return SessionListResponse(
        sessions=session_ids,
        count=len(session_ids),
    )


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_session() -> dict:
    """Create a new empty session."""
    session_id = str(uuid4())
    session_memory.get_memory(session_id)
    logger.info(f"Created new session: {session_id}")
    return {"session_id": session_id}


@router.get(
    "/{session_id}",
    response_model=SessionHistoryResponse,
    status_code=status.HTTP_200_OK,
)
async def get_session_history(session_id: str) -> SessionHistoryResponse:
    """Get conversation history for a session."""
    logger.info(f"Retrieving history for session: {session_id}")

    if not session_memory.session_exists(session_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found",
        )

    try:
        history = session_memory.get_history(session_id)

        messages = [
            MessageInfo(
                role="human" if "HumanMessage" in type(msg).__name__ else "ai",
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


@router.delete("/{session_id}", status_code=status.HTTP_200_OK)
async def clear_session(session_id: str) -> ClearMemoryResponse:
    """Clear conversation memory for a session."""
    logger.info(f"Clearing memory for session: {session_id}")

    try:
        success = session_memory.clear_memory(session_id)

        if success:
            return ClearMemoryResponse(
                success=True,
                message=f"Successfully cleared memory for session {session_id}",
            )
        else:
            return ClearMemoryResponse(
                success=False,
                message=f"Session {session_id} not found or already empty",
            )

    except Exception as e:
        logger.exception(f"Error clearing memory: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear memory: {str(e)}",
        )


@router.delete("/{session_id}/permanently", status_code=status.HTTP_200_OK)
async def delete_session(session_id: str) -> dict:
    """Permanently delete a session."""
    logger.info(f"Deleting session: {session_id}")

    success = session_memory.delete_session(session_id)

    if success:
        return {"message": f"Session {session_id} deleted permanently"}
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found",
        )
