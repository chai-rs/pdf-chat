"""Database module for vector store and session memory."""

from app.db.memory import (
    add_message,
    clear_memory,
    delete_session,
    get_history,
    get_memory,
    get_session_ids,
    session_exists,
)
from app.db.session_memory import SessionMemory
from app.db.vector_store import (
    VectorStore,
    get_chroma_client,
    get_retriever,
    get_vector_store,
    is_result_relevant,
    search_documents,
)

__all__ = [
    # Vector store
    "VectorStore",
    "get_chroma_client",
    "get_vector_store",
    "get_retriever",
    "search_documents",
    "is_result_relevant",
    # Session memory (legacy)
    "SessionMemory",
    # Conversation memory
    "get_memory",
    "add_message",
    "get_history",
    "clear_memory",
    "delete_session",
    "get_session_ids",
    "session_exists",
]
