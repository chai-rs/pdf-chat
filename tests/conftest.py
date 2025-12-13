"""Pytest fixtures for Chat with PDF API tests."""

from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.core.dependencies import get_vector_store


# --- Mock Vector Store ---


class MockVectorStore:
    """Mock vector store for testing."""

    def __init__(self):
        self._documents: list[dict] = []

    @property
    def vectorstore(self) -> MagicMock:
        """Return a mock vectorstore."""
        return MagicMock()

    def add_documents(self, documents: list[Any], metadatas: list[dict] | None = None) -> list[str]:
        """Mock add documents."""
        ids = [f"doc_{i}" for i in range(len(documents))]
        for i, doc in enumerate(documents):
            self._documents.append(
                {
                    "id": ids[i],
                    "content": doc
                    if isinstance(doc, str)
                    else getattr(doc, "page_content", str(doc)),
                    "metadata": metadatas[i] if metadatas else {},
                }
            )
        return ids

    def similarity_search_with_score(self, query: str, k: int = 4) -> list[tuple[Any, float]]:
        """Mock similarity search returning sample documents."""
        # Return mock documents with relevance scores
        mock_results = []
        for i, doc in enumerate(self._documents[:k]):
            mock_doc = MagicMock()
            mock_doc.page_content = doc.get("content", f"Mock content for query: {query}")
            mock_doc.metadata = doc.get("metadata", {"source": "test.pdf", "page": i + 1})
            mock_results.append((mock_doc, 0.85 - (i * 0.1)))

        # If no documents, return default mock results
        if not mock_results:
            for i in range(min(k, 2)):
                mock_doc = MagicMock()
                mock_doc.page_content = "This research explores text-to-SQL approaches. SimpleDDL-MD-Chat achieves highest zero-shot accuracy on Spider benchmark (Zhang et al., 2024)."
                mock_doc.metadata = {"source": "text-to-sql-survey.pdf", "page": i + 1}
                mock_results.append((mock_doc, 0.9 - (i * 0.1)))

        return mock_results

    def delete_collection(self) -> bool:
        """Mock delete collection."""
        self._documents = []
        return True


# --- Fixtures ---


@pytest.fixture
def mock_vector_store() -> MockVectorStore:
    """Provide a mock vector store instance."""
    return MockVectorStore()


@pytest.fixture
def client(mock_vector_store: MockVectorStore) -> TestClient:
    """Provide a test client with mocked dependencies."""
    app.dependency_overrides[get_vector_store] = lambda: mock_vector_store
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
async def async_client(mock_vector_store: MockVectorStore) -> AsyncGenerator[AsyncClient, None]:
    """Provide an async test client with mocked dependencies."""
    app.dependency_overrides[get_vector_store] = lambda: mock_vector_store
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()


@pytest.fixture
def mock_llm() -> MagicMock:
    """Provide a mock LLM that returns predictable responses."""
    mock = MagicMock()
    mock.invoke = MagicMock(return_value=MagicMock(content="Mock LLM response"))
    mock.ainvoke = AsyncMock(return_value=MagicMock(content="Mock LLM async response"))
    return mock


@pytest.fixture
def mock_orchestrator_clarification() -> dict:
    """Mock orchestrator result requesting clarification."""
    return {
        "needs_clarification": True,
        "clarification_question": "Could you please specify which topic you're referring to?",
        "route": "clarification",
        "final_answer": "",
        "pdf_results": [],
        "web_results": [],
    }


@pytest.fixture
def mock_orchestrator_pdf_result() -> dict:
    """Mock orchestrator result from PDF search."""
    return {
        "needs_clarification": False,
        "route": "pdf_search",
        "final_answer": "According to the research, SimpleDDL-MD-Chat achieves the highest zero-shot accuracy on the Spider benchmark (Zhang et al., 2024).",
        "pdf_results": [
            {
                "content": "SimpleDDL-MD-Chat prompt template achieved highest zero-shot accuracy on Spider dataset.",
                "metadata": {"source": "text-to-sql-survey.pdf", "page": 12},
            }
        ],
        "web_results": [],
    }


@pytest.fixture
def mock_orchestrator_web_result() -> dict:
    """Mock orchestrator result from web search."""
    return {
        "needs_clarification": False,
        "route": "web_search",
        "final_answer": "Based on recent developments, OpenAI released GPT-4 Turbo with improved capabilities.",
        "pdf_results": [],
        "web_results": [
            {
                "title": "OpenAI Releases GPT-4 Turbo",
                "url": "https://openai.com/blog/new-models",
                "snippet": "OpenAI announces GPT-4 Turbo with improved performance.",
            }
        ],
    }


@pytest.fixture
def mock_orchestrator_hybrid_result() -> dict:
    """Mock orchestrator result from hybrid search."""
    return {
        "needs_clarification": False,
        "route": "hybrid",
        "final_answer": "State-of-the-art text-to-SQL systems combine LLM prompting with schema linking. Recent work by Zhang et al. (2024) shows SimpleDDL achieving top results.",
        "pdf_results": [
            {
                "content": "Text-to-SQL state-of-the-art approaches use LLM with schema information.",
                "metadata": {"source": "survey.pdf", "page": 5},
            }
        ],
        "web_results": [
            {
                "title": "Latest Text-to-SQL Research",
                "url": "https://arxiv.org/latest",
                "snippet": "Recent advances in text-to-SQL methodologies.",
            }
        ],
    }


# --- Session Fixtures ---


@pytest.fixture
def sample_session_id() -> str:
    """Provide a sample session ID for testing."""
    return "test-session-12345"


@pytest.fixture(autouse=True)
def clear_session_memory():
    """Clear session memory before each test."""
    from app.db import memory as session_memory

    # Clear any existing sessions
    for session_id in list(session_memory.get_session_ids()):
        session_memory.delete_session(session_id)

    yield

    # Cleanup after test
    for session_id in list(session_memory.get_session_ids()):
        session_memory.delete_session(session_id)
