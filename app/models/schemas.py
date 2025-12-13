"""Pydantic schemas for API requests and responses."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    """Chat message model."""

    role: Literal["human", "ai", "system"] = Field(description="Message role")
    content: str = Field(description="Message content")


# --- Chat Models ---


class DocumentSource(BaseModel):
    """Source document from vector store."""

    content: str = Field(description="Document content snippet")
    metadata: dict = Field(default_factory=dict, description="Document metadata")


class WebSource(BaseModel):
    """Web search result source."""

    title: str = Field(default="", description="Web page title")
    url: str = Field(default="", description="Web page URL")
    content: str = Field(default="", description="Content snippet")


class ChatRequest(BaseModel):
    """Chat request model."""

    message: str = Field(description="User message/question")
    session_id: str | None = Field(default=None, description="Session ID for conversation history")
    use_web_search: bool = Field(
        default=False, description="Enable web search for additional context"
    )


class ChatResponse(BaseModel):
    """Chat response model."""

    response: str = Field(description="AI response")
    session_id: str = Field(description="Session ID for this conversation")
    sources: list[DocumentSource] = Field(default_factory=list, description="Source documents used")
    web_sources: list[WebSource] = Field(
        default_factory=list, description="Web search sources used"
    )
    needs_clarification: bool = Field(
        default=False, description="Whether the query needs clarification"
    )
    clarification_question: str | None = Field(
        default=None, description="Clarification question if needed"
    )


class ClearMemoryRequest(BaseModel):
    """Request to clear session memory."""

    session_id: str = Field(description="Session ID to clear")


# --- Document Models ---


class DocumentInfo(BaseModel):
    """Document information model."""

    filename: str = Field(description="Original filename")
    file_hash: str = Field(description="MD5 hash of the file")
    num_chunks: int = Field(description="Number of chunks created")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Upload timestamp")


class DocumentUploadResponse(BaseModel):
    """Document upload response model."""

    success: bool = Field(description="Upload success status")
    message: str = Field(description="Status message")
    document: DocumentInfo | None = Field(default=None, description="Document info if successful")


# --- Health & Session Models ---


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str = Field(description="Service status")
    version: str = Field(default="0.1.0", description="API version")


class SessionInfo(BaseModel):
    """Session information model."""

    session_id: str = Field(description="Session ID")
    message_count: int = Field(description="Number of messages in session")
    messages: list[Message] = Field(default_factory=list, description="Session messages")


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str = Field(description="Error message")
    detail: str | None = Field(default=None, description="Detailed error information")
