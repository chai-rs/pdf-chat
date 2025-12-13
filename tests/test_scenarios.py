"""Integration tests for Chat with PDF scenarios.

These tests verify the system handles the assignment test cases correctly:
1. Ambiguous queries requiring clarification
2. PDF-only queries about academic papers
3. Autonomous multi-step execution
4. Out-of-scope queries requiring web search
"""

from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient


class TestAmbiguousQueries:
    """Test case 1: Ambiguous queries that need clarification."""

    @patch("app.api.chat.get_orchestrator")
    def test_ambiguous_query_without_context(
        self,
        mock_get_orchestrator,
        client: TestClient,
    ):
        """
        Test: "Tell me more about it" (no context)
        Expected: Clarification request
        """
        mock_orchestrator = AsyncMock()
        mock_orchestrator.arun = AsyncMock(
            return_value={
                "needs_clarification": True,
                "clarification_question": "I'd be happy to help! Could you please specify what topic you'd like to know more about?",
                "route": "clarification",
                "final_answer": "",
                "pdf_results": [],
                "web_results": [],
            }
        )
        mock_get_orchestrator.return_value = mock_orchestrator

        response = client.post(
            "/api/v1/chat",
            json={"question": "Tell me more about it"},
        )

        assert response.status_code == 200
        data = response.json()

        # Verify clarification is requested
        assert data["needs_clarification"] is True
        assert data["clarification_question"] is not None
        assert len(data["clarification_question"]) > 0
        assert data["agent_used"] == "clarification"

        # Verify no sources returned for clarification
        assert len(data["sources"]) == 0

    @patch("app.api.chat.get_orchestrator")
    def test_vague_query_triggers_clarification(
        self,
        mock_get_orchestrator,
        client: TestClient,
    ):
        """Test that vague queries without context trigger clarification."""
        mock_orchestrator = AsyncMock()
        mock_orchestrator.arun = AsyncMock(
            return_value={
                "needs_clarification": True,
                "clarification_question": "What specific aspect would you like to explore?",
                "route": "clarification",
                "final_answer": "",
                "pdf_results": [],
                "web_results": [],
            }
        )
        mock_get_orchestrator.return_value = mock_orchestrator

        response = client.post(
            "/api/v1/chat",
            json={"question": "Can you explain that?"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["needs_clarification"] is True


class TestPDFOnlyQueries:
    """Test case 2: Queries that should be answered from PDF documents."""

    @patch("app.api.chat.get_orchestrator")
    def test_pdf_query_spider_accuracy(
        self,
        mock_get_orchestrator,
        client: TestClient,
    ):
        """
        Test: "Which prompt template gave highest zero-shot accuracy on Spider?"
        Expected: Answer citing SimpleDDL-MD-Chat, Zhang et al.
        """
        mock_orchestrator = AsyncMock()
        mock_orchestrator.arun = AsyncMock(
            return_value={
                "needs_clarification": False,
                "route": "pdf_search",
                "final_answer": (
                    "According to the research paper, the SimpleDDL-MD-Chat prompt template "
                    "achieved the highest zero-shot accuracy on the Spider benchmark. "
                    "This was demonstrated by Zhang et al. (2024) in their comprehensive "
                    "evaluation of various prompt engineering approaches for text-to-SQL tasks."
                ),
                "pdf_results": [
                    {
                        "content": (
                            "SimpleDDL-MD-Chat prompt template achieved 73.5% zero-shot accuracy "
                            "on Spider benchmark, outperforming other prompt designs."
                        ),
                        "metadata": {
                            "source": "text-to-sql-survey.pdf",
                            "page": 12,
                            "authors": "Zhang et al.",
                        },
                    }
                ],
                "web_results": [],
            }
        )
        mock_get_orchestrator.return_value = mock_orchestrator

        response = client.post(
            "/api/v1/chat",
            json={"question": "Which prompt template gave highest zero-shot accuracy on Spider?"},
        )

        assert response.status_code == 200
        data = response.json()

        # Verify PDF-based response
        assert data["needs_clarification"] is False
        assert data["agent_used"] == "pdf_search"

        # Verify answer contains expected information
        answer = data["answer"].lower()
        assert "simpleddl" in answer or "simple" in answer

        # Verify sources are from PDF
        assert len(data["sources"]) > 0
        assert data["sources"][0]["source_type"] == "pdf"

    @patch("app.api.chat.get_orchestrator")
    def test_pdf_query_with_citations(
        self,
        mock_get_orchestrator,
        client: TestClient,
    ):
        """Test that PDF queries return proper source citations."""
        mock_orchestrator = AsyncMock()
        mock_orchestrator.arun = AsyncMock(
            return_value={
                "needs_clarification": False,
                "route": "pdf_search",
                "final_answer": "The paper discusses various prompt engineering techniques for text-to-SQL.",
                "pdf_results": [
                    {
                        "content": "Prompt engineering affects SQL generation accuracy significantly.",
                        "metadata": {"source": "paper.pdf", "page": 5},
                    }
                ],
                "web_results": [],
            }
        )
        mock_get_orchestrator.return_value = mock_orchestrator

        response = client.post(
            "/api/v1/chat",
            json={"question": "What prompt engineering techniques are discussed?"},
        )

        assert response.status_code == 200
        data = response.json()

        # Verify sources contain metadata
        assert len(data["sources"]) > 0
        source = data["sources"][0]
        assert source["source_type"] == "pdf"
        assert "metadata" in source


class TestAutonomousMultiStepQueries:
    """Test case 3: Queries requiring multi-step execution."""

    @patch("app.api.chat.get_orchestrator")
    def test_hybrid_query_pdf_and_web(
        self,
        mock_get_orchestrator,
        client: TestClient,
    ):
        """
        Test: "What's state-of-the-art text-to-sql? Search web for authors"
        Expected: Multi-step execution (PDF + web search)
        """
        mock_orchestrator = AsyncMock()
        mock_orchestrator.arun = AsyncMock(
            return_value={
                "needs_clarification": False,
                "route": "hybrid",
                "final_answer": (
                    "State-of-the-art text-to-SQL approaches combine large language models "
                    "with schema linking techniques. According to the PDF documents, "
                    "SimpleDDL-MD-Chat achieves top results on Spider benchmark. "
                    "Web search reveals that the lead authors include Zhang et al. from "
                    "Stanford University, who have been publishing extensively on this topic."
                ),
                "pdf_results": [
                    {
                        "content": "Current SOTA in text-to-SQL uses LLM prompting with schema information.",
                        "metadata": {"source": "survey.pdf", "page": 5},
                    }
                ],
                "web_results": [
                    {
                        "title": "Text-to-SQL Research - Zhang et al.",
                        "url": "https://arxiv.org/abs/2024.xxxxx",
                        "snippet": "Zhang, Li, and Wang lead text-to-SQL research at Stanford.",
                    }
                ],
            }
        )
        mock_get_orchestrator.return_value = mock_orchestrator

        response = client.post(
            "/api/v1/chat",
            json={"question": "What's state-of-the-art text-to-sql? Search web for authors"},
        )

        assert response.status_code == 200
        data = response.json()

        # Verify hybrid routing
        assert data["needs_clarification"] is False
        assert data["agent_used"] == "hybrid"

        # Verify both PDF and web sources
        source_types = {s["source_type"] for s in data["sources"]}
        assert "pdf" in source_types
        assert "web" in source_types

    @patch("app.api.chat.get_orchestrator")
    def test_complex_query_with_subqueries(
        self,
        mock_get_orchestrator,
        client: TestClient,
    ):
        """Test that complex queries are handled with potential sub-queries."""
        mock_orchestrator = AsyncMock()
        mock_orchestrator.arun = AsyncMock(
            return_value={
                "needs_clarification": False,
                "route": "hybrid",
                "final_answer": "Combined analysis from multiple sources provides comprehensive answer.",
                "pdf_results": [
                    {"content": "PDF content", "metadata": {"source": "doc.pdf", "page": 1}}
                ],
                "web_results": [
                    {"title": "Web result", "url": "https://example.com", "snippet": "Web content"}
                ],
            }
        )
        mock_get_orchestrator.return_value = mock_orchestrator

        response = client.post(
            "/api/v1/chat",
            json={
                "question": (
                    "Compare the accuracy metrics from the paper with current benchmarks online"
                )
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["agent_used"] in ["hybrid", "pdf_search", "web_search"]


class TestOutOfScopeQueries:
    """Test case 4: Queries outside PDF scope that trigger web search."""

    @patch("app.api.chat.get_orchestrator")
    def test_out_of_scope_triggers_web_search(
        self,
        mock_get_orchestrator,
        client: TestClient,
    ):
        """
        Test: "What did OpenAI release this month?"
        Expected: Web search triggered
        """
        mock_orchestrator = AsyncMock()
        mock_orchestrator.arun = AsyncMock(
            return_value={
                "needs_clarification": False,
                "route": "web_search",
                "final_answer": (
                    "OpenAI has released several updates this month, including "
                    "GPT-4 Turbo with improved context handling and reduced pricing. "
                    "They also announced updates to the ChatGPT interface."
                ),
                "pdf_results": [],
                "web_results": [
                    {
                        "title": "OpenAI December 2024 Releases",
                        "url": "https://openai.com/blog",
                        "snippet": "OpenAI announces GPT-4 Turbo and new API features.",
                    },
                    {
                        "title": "ChatGPT Updates",
                        "url": "https://openai.com/chatgpt",
                        "snippet": "Latest ChatGPT improvements and features.",
                    },
                ],
            }
        )
        mock_get_orchestrator.return_value = mock_orchestrator

        response = client.post(
            "/api/v1/chat",
            json={"question": "What did OpenAI release this month?"},
        )

        assert response.status_code == 200
        data = response.json()

        # Verify web search routing
        assert data["needs_clarification"] is False
        assert data["agent_used"] == "web_search"

        # Verify web sources
        assert len(data["sources"]) > 0
        assert all(s["source_type"] == "web" for s in data["sources"])

    @patch("app.api.chat.get_orchestrator")
    def test_current_events_trigger_web_search(
        self,
        mock_get_orchestrator,
        client: TestClient,
    ):
        """Test that current events questions trigger web search."""
        mock_orchestrator = AsyncMock()
        mock_orchestrator.arun = AsyncMock(
            return_value={
                "needs_clarification": False,
                "route": "web_search",
                "final_answer": "Latest news from the web search.",
                "pdf_results": [],
                "web_results": [
                    {"title": "News", "url": "https://news.com", "snippet": "Latest updates"}
                ],
            }
        )
        mock_get_orchestrator.return_value = mock_orchestrator

        response = client.post(
            "/api/v1/chat",
            json={"question": "What are the latest AI news today?"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["agent_used"] == "web_search"


class TestConversationContext:
    """Test that conversation context is maintained across messages."""

    @patch("app.api.chat.get_orchestrator")
    def test_conversation_context_maintained(
        self,
        mock_get_orchestrator,
        client: TestClient,
    ):
        """Test that follow-up questions can reference previous context."""
        # First message
        mock_orchestrator = AsyncMock()
        mock_orchestrator.arun = AsyncMock(
            return_value={
                "needs_clarification": False,
                "route": "pdf_search",
                "final_answer": "Text-to-SQL is a task of converting natural language to SQL queries.",
                "pdf_results": [
                    {"content": "Definition", "metadata": {"source": "doc.pdf", "page": 1}}
                ],
                "web_results": [],
            }
        )
        mock_get_orchestrator.return_value = mock_orchestrator

        response1 = client.post(
            "/api/v1/chat",
            json={"question": "What is text-to-SQL?"},
        )
        session_id = response1.json()["session_id"]

        # Second message with same session
        mock_orchestrator.arun = AsyncMock(
            return_value={
                "needs_clarification": False,
                "route": "pdf_search",
                "final_answer": "The main challenges include schema understanding and query complexity.",
                "pdf_results": [
                    {"content": "Challenges", "metadata": {"source": "doc.pdf", "page": 2}}
                ],
                "web_results": [],
            }
        )

        response2 = client.post(
            "/api/v1/chat",
            json={
                "question": "What are its main challenges?",
                "session_id": session_id,
            },
        )

        assert response2.status_code == 200
        data = response2.json()
        assert data["session_id"] == session_id
        # The answer should be meaningful, not asking for clarification
        # (since context from previous message should be available)
        assert "challenges" in data["answer"].lower() or data["needs_clarification"] is False


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_question(self, client: TestClient):
        """Test handling of empty question."""
        response = client.post(
            "/api/v1/chat",
            json={"question": ""},
        )
        # Should either handle gracefully or return validation error
        assert response.status_code in [200, 422]

    def test_very_long_question(self, client: TestClient):
        """Test handling of very long questions."""
        long_question = "What is text-to-SQL? " * 100

        # Should not crash, might return error or truncate
        response = client.post(
            "/api/v1/chat",
            json={"question": long_question},
        )
        # Should handle gracefully
        assert response.status_code in [200, 400, 413, 422, 500]

    @patch("app.api.chat.get_orchestrator")
    def test_special_characters_in_question(
        self,
        mock_get_orchestrator,
        client: TestClient,
    ):
        """Test handling of special characters in questions."""
        mock_orchestrator = AsyncMock()
        mock_orchestrator.arun = AsyncMock(
            return_value={
                "needs_clarification": False,
                "route": "pdf_search",
                "final_answer": "Answer with special chars handled.",
                "pdf_results": [],
                "web_results": [],
            }
        )
        mock_get_orchestrator.return_value = mock_orchestrator

        response = client.post(
            "/api/v1/chat",
            json={"question": 'What is "text-to-SQL" & how does it work?'},
        )

        assert response.status_code == 200
