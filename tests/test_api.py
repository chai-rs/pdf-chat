"""API endpoint tests for Chat with PDF."""

from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient


class TestHealthEndpoint:
    """Tests for the health check endpoint."""

    def test_health_check_returns_healthy(self, client: TestClient):
        """Test that health endpoint returns healthy status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "vector_store" in data

    def test_root_endpoint_returns_api_info(self, client: TestClient):
        """Test that root endpoint returns API information."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Chat with PDF API"
        assert data["version"] == "1.0.0"
        assert "docs" in data
        assert "health" in data


class TestChatEndpoint:
    """Tests for the chat endpoint."""

    @patch("app.api.chat.get_orchestrator")
    def test_chat_with_clarification_needed(
        self,
        mock_get_orchestrator,
        client: TestClient,
        mock_orchestrator_clarification: dict,
    ):
        """Test chat endpoint when clarification is needed."""
        mock_orchestrator = AsyncMock()
        mock_orchestrator.arun = AsyncMock(return_value=mock_orchestrator_clarification)
        mock_get_orchestrator.return_value = mock_orchestrator

        response = client.post(
            "/api/v1/chat",
            json={"question": "Tell me more about it"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["needs_clarification"] is True
        assert data["clarification_question"] is not None
        assert data["agent_used"] == "clarification"
        assert "session_id" in data

    @patch("app.api.chat.get_orchestrator")
    def test_chat_with_pdf_response(
        self,
        mock_get_orchestrator,
        client: TestClient,
        mock_orchestrator_pdf_result: dict,
    ):
        """Test chat endpoint returning PDF-based answer."""
        mock_orchestrator = AsyncMock()
        mock_orchestrator.arun = AsyncMock(return_value=mock_orchestrator_pdf_result)
        mock_get_orchestrator.return_value = mock_orchestrator

        response = client.post(
            "/api/v1/chat",
            json={"question": "Which prompt template gave highest zero-shot accuracy on Spider?"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["needs_clarification"] is False
        assert "SimpleDDL" in data["answer"] or len(data["answer"]) > 0
        assert data["agent_used"] == "pdf_search"
        assert len(data["sources"]) > 0
        assert data["sources"][0]["source_type"] == "pdf"

    @patch("app.api.chat.get_orchestrator")
    def test_chat_with_web_response(
        self,
        mock_get_orchestrator,
        client: TestClient,
        mock_orchestrator_web_result: dict,
    ):
        """Test chat endpoint returning web search answer."""
        mock_orchestrator = AsyncMock()
        mock_orchestrator.arun = AsyncMock(return_value=mock_orchestrator_web_result)
        mock_get_orchestrator.return_value = mock_orchestrator

        response = client.post(
            "/api/v1/chat",
            json={"question": "What did OpenAI release this month?"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["needs_clarification"] is False
        assert data["agent_used"] == "web_search"
        assert len(data["sources"]) > 0
        assert data["sources"][0]["source_type"] == "web"

    @patch("app.api.chat.get_orchestrator")
    def test_chat_with_session_id(
        self,
        mock_get_orchestrator,
        client: TestClient,
        mock_orchestrator_pdf_result: dict,
        sample_session_id: str,
    ):
        """Test chat endpoint with provided session ID."""
        mock_orchestrator = AsyncMock()
        mock_orchestrator.arun = AsyncMock(return_value=mock_orchestrator_pdf_result)
        mock_get_orchestrator.return_value = mock_orchestrator

        response = client.post(
            "/api/v1/chat",
            json={
                "question": "What is text-to-SQL?",
                "session_id": sample_session_id,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == sample_session_id

    @patch("app.api.chat.get_orchestrator")
    def test_chat_creates_session_if_not_provided(
        self,
        mock_get_orchestrator,
        client: TestClient,
        mock_orchestrator_pdf_result: dict,
    ):
        """Test that chat creates a new session if none is provided."""
        mock_orchestrator = AsyncMock()
        mock_orchestrator.arun = AsyncMock(return_value=mock_orchestrator_pdf_result)
        mock_get_orchestrator.return_value = mock_orchestrator

        response = client.post(
            "/api/v1/chat",
            json={"question": "What is text-to-SQL?"},
        )

        assert response.status_code == 200
        data = response.json()
        # Should have a valid UUID session_id
        assert len(data["session_id"]) == 36
        assert data["session_id"].count("-") == 4

    @patch("app.api.chat.get_orchestrator")
    def test_chat_handles_orchestrator_error(
        self,
        mock_get_orchestrator,
        client: TestClient,
    ):
        """Test chat endpoint handles orchestrator errors gracefully."""
        mock_orchestrator = AsyncMock()
        mock_orchestrator.arun = AsyncMock(side_effect=Exception("LLM connection failed"))
        mock_get_orchestrator.return_value = mock_orchestrator

        response = client.post(
            "/api/v1/chat",
            json={"question": "Test question"},
        )

        assert response.status_code == 500
        data = response.json()
        assert "Failed to process chat request" in data["detail"]


class TestClearMemory:
    """Tests for the clear memory/session endpoints."""

    def test_clear_session_memory(self, client: TestClient):
        """Test clearing session memory."""
        # First create a session
        response = client.post("/api/v1/sessions")
        assert response.status_code == 201
        session_id = response.json()["session_id"]

        # Clear the session memory
        response = client.delete(f"/api/v1/sessions/{session_id}")
        assert response.status_code == 200
        data = response.json()
        # Session was just created, so it's empty
        assert "success" in data

    def test_clear_nonexistent_session(self, client: TestClient):
        """Test clearing a session that doesn't exist."""
        response = client.delete("/api/v1/sessions/nonexistent-session-id")
        assert response.status_code == 200
        data = response.json()
        # Should indicate session not found but not error
        assert data["success"] is False or "not found" in data["message"]

    def test_delete_session_permanently(self, client: TestClient):
        """Test permanently deleting a session."""
        # Create a session
        response = client.post("/api/v1/sessions")
        assert response.status_code == 201
        session_id = response.json()["session_id"]

        # Delete permanently
        response = client.delete(f"/api/v1/sessions/{session_id}/permanently")
        assert response.status_code == 200
        data = response.json()
        assert "deleted permanently" in data["message"]

        # Session should no longer exist
        response = client.get(f"/api/v1/sessions/{session_id}")
        assert response.status_code == 404


class TestSessionManagement:
    """Tests for session management endpoints."""

    def test_list_sessions(self, client: TestClient):
        """Test listing all sessions."""
        response = client.get("/api/v1/sessions")
        assert response.status_code == 200
        data = response.json()
        assert "sessions" in data
        assert "count" in data
        assert isinstance(data["sessions"], list)

    def test_create_session(self, client: TestClient):
        """Test creating a new session."""
        response = client.post("/api/v1/sessions")
        assert response.status_code == 201
        data = response.json()
        assert "session_id" in data
        assert len(data["session_id"]) == 36

    def test_get_session_history(self, client: TestClient):
        """Test getting session history."""
        # Create a session
        response = client.post("/api/v1/sessions")
        session_id = response.json()["session_id"]

        # Get history
        response = client.get(f"/api/v1/sessions/{session_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == session_id
        assert "messages" in data
        assert "message_count" in data

    def test_get_nonexistent_session_history(self, client: TestClient):
        """Test getting history for a nonexistent session."""
        response = client.get("/api/v1/sessions/nonexistent-session")
        assert response.status_code == 404
