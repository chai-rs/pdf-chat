"""Session memory for conversation history."""

from collections import defaultdict
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


class SessionMemory:
    """In-memory session storage for conversation history."""

    def __init__(self):
        """Initialize session memory."""
        self._sessions: dict[str, list[BaseMessage]] = defaultdict(list)

    def add_message(self, session_id: str, message: BaseMessage) -> None:
        """Add a message to a session.

        Args:
            session_id: Unique session identifier
            message: LangChain message to add
        """
        self._sessions[session_id].append(message)

    def add_human_message(self, session_id: str, content: str) -> None:
        """Add a human message to a session.

        Args:
            session_id: Unique session identifier
            content: Message content
        """
        self.add_message(session_id, HumanMessage(content=content))

    def add_ai_message(self, session_id: str, content: str) -> None:
        """Add an AI message to a session.

        Args:
            session_id: Unique session identifier
            content: Message content
        """
        self.add_message(session_id, AIMessage(content=content))

    def get_messages(self, session_id: str) -> list[BaseMessage]:
        """Get all messages for a session.

        Args:
            session_id: Unique session identifier

        Returns:
            List of messages
        """
        return self._sessions[session_id].copy()

    def get_recent_messages(self, session_id: str, n: int = 10) -> list[BaseMessage]:
        """Get the most recent messages for a session.

        Args:
            session_id: Unique session identifier
            n: Number of recent messages to return

        Returns:
            List of recent messages
        """
        return self._sessions[session_id][-n:]

    def clear_session(self, session_id: str) -> None:
        """Clear all messages for a session.

        Args:
            session_id: Unique session identifier
        """
        self._sessions[session_id] = []

    def delete_session(self, session_id: str) -> None:
        """Delete a session entirely.

        Args:
            session_id: Unique session identifier
        """
        if session_id in self._sessions:
            del self._sessions[session_id]

    def get_session_ids(self) -> list[str]:
        """Get all active session IDs.

        Returns:
            List of session IDs
        """
        return list(self._sessions.keys())

    def to_dict(self, session_id: str) -> list[dict[str, Any]]:
        """Convert session messages to dictionary format.

        Args:
            session_id: Unique session identifier

        Returns:
            List of message dictionaries
        """
        return [
            {"role": "human" if isinstance(m, HumanMessage) else "ai", "content": m.content}
            for m in self._sessions[session_id]
        ]


# Global session memory instance
session_memory = SessionMemory()
