"""Session-based conversation memory with ConversationBufferMemory integration."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING
from uuid import uuid4

from langchain_classic.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

if TYPE_CHECKING:
    pass


# Thread-safe lock for concurrent access
_memory_lock = threading.RLock()

# In-memory storage: session_id -> ChatMessageHistory
_sessions: dict[str, ChatMessageHistory] = {}

# Maximum number of message pairs to retain (10 pairs = 20 messages)
MAX_MESSAGE_PAIRS = 10


def _get_or_create_session(session_id: str | None = None) -> tuple[str, ChatMessageHistory]:
    """Get existing or create new session with ChatMessageHistory.

    Args:
        session_id: Optional session identifier. If None, generates a new UUID.

    Returns:
        Tuple of (session_id, ChatMessageHistory)
    """
    if session_id is None:
        session_id = str(uuid4())

    with _memory_lock:
        if session_id not in _sessions:
            _sessions[session_id] = ChatMessageHistory()
        return session_id, _sessions[session_id]


def _truncate_history(history: ChatMessageHistory) -> None:
    """Truncate history to keep only the last MAX_MESSAGE_PAIRS pairs.

    This prevents context overflow by limiting the conversation history.
    A message pair consists of one human message and one AI response.

    Args:
        history: ChatMessageHistory to truncate in-place
    """
    max_messages = MAX_MESSAGE_PAIRS * 2  # 10 pairs = 20 messages

    if len(history.messages) > max_messages:
        # Keep only the last max_messages
        history.messages = history.messages[-max_messages:]


def get_memory(session_id: str | None = None) -> tuple[str, ConversationBufferMemory]:
    """Get ConversationBufferMemory for a session.

    Creates a new session if the session_id doesn't exist or is None.

    Args:
        session_id: Optional session identifier. Auto-generates UUID if not provided.

    Returns:
        Tuple of (session_id, ConversationBufferMemory) - includes session_id in case
        it was auto-generated.
    """
    session_id, history = _get_or_create_session(session_id)

    with _memory_lock:
        memory = ConversationBufferMemory(
            chat_memory=history,
            return_messages=True,
            memory_key="chat_history",
        )
        return session_id, memory


def add_message(session_id: str | None, role: str, content: str) -> str:
    """Add a message to the session history.

    Args:
        session_id: Session identifier. Auto-generates UUID if None.
        role: Message role - "human", "user", "ai", or "assistant"
        content: Message content

    Returns:
        The session_id (useful when auto-generated)

    Raises:
        ValueError: If role is not recognized
    """
    session_id, history = _get_or_create_session(session_id)

    with _memory_lock:
        role_lower = role.lower()
        if role_lower in ("human", "user"):
            history.add_message(HumanMessage(content=content))
        elif role_lower in ("ai", "assistant"):
            history.add_message(AIMessage(content=content))
        else:
            raise ValueError(
                f"Unknown role: {role}. Expected 'human', 'user', 'ai', or 'assistant'"
            )

        # Truncate to prevent context overflow
        _truncate_history(history)

    return session_id


def get_history(session_id: str) -> list[BaseMessage]:
    """Get conversation history for a session.

    Args:
        session_id: Session identifier

    Returns:
        List of BaseMessage objects. Returns empty list if session doesn't exist.
    """
    with _memory_lock:
        if session_id not in _sessions:
            return []
        return list(_sessions[session_id].messages)


def clear_memory(session_id: str) -> bool:
    """Clear all messages for a session.

    Args:
        session_id: Session identifier

    Returns:
        True if session existed and was cleared, False if session didn't exist
    """
    with _memory_lock:
        if session_id in _sessions:
            _sessions[session_id].clear()
            return True
        return False


def delete_session(session_id: str) -> bool:
    """Delete a session entirely from memory.

    Args:
        session_id: Session identifier

    Returns:
        True if session existed and was deleted, False if session didn't exist
    """
    with _memory_lock:
        if session_id in _sessions:
            del _sessions[session_id]
            return True
        return False


def get_session_ids() -> list[str]:
    """Get all active session IDs.

    Returns:
        List of session IDs currently in memory
    """
    with _memory_lock:
        return list(_sessions.keys())


def session_exists(session_id: str) -> bool:
    """Check if a session exists.

    Args:
        session_id: Session identifier

    Returns:
        True if session exists, False otherwise
    """
    with _memory_lock:
        return session_id in _sessions
