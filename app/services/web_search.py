"""Web search service using Tavily API with fallback support."""

import logging
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# --- Pydantic Models ---


class SearchResult(BaseModel):
    """Web search result model."""

    title: str = Field(default="", description="Page title")
    url: str = Field(default="", description="Page URL")
    snippet: str = Field(default="", description="Content snippet/summary")
    relevance_score: float = Field(default=0.0, description="Relevance score (0.0-1.0)")


# --- Mock Data for Testing ---


def _get_mock_results(query: str) -> list[SearchResult]:
    """Get mock search results for testing when API is unavailable.

    Args:
        query: Search query (used to customize mock results)

    Returns:
        List of mock SearchResult objects
    """
    return [
        SearchResult(
            title=f"Mock Result 1 for: {query}",
            url="https://example.com/result1",
            snippet=f"This is a mock search result about '{query}'. Use this for testing when Tavily API is unavailable.",
            relevance_score=0.95,
        ),
        SearchResult(
            title=f"Mock Result 2 for: {query}",
            url="https://example.com/result2",
            snippet=f"Another mock result related to '{query}'. Contains sample information for development.",
            relevance_score=0.85,
        ),
        SearchResult(
            title=f"Mock Result 3 for: {query}",
            url="https://example.com/result3",
            snippet=f"Third mock result about '{query}'. Simulates web search functionality.",
            relevance_score=0.75,
        ),
    ]


# --- Tavily Search Service ---


class TavilySearchService:
    """Service for web search using Tavily API.

    Features:
        - Search with configurable max results
        - Format results for LLM context
        - Fallback to mock data for testing
        - Supports langchain_community.tools.TavilySearchResults if available

    Environment variable: TAVILY_API_KEY
    """

    def __init__(self, api_key: str):
        """Initialize the Tavily search service.

        Args:
            api_key: Tavily API key (from TAVILY_API_KEY env var)
        """
        self.api_key = api_key
        self._client: Any = None
        self._langchain_tool: Any = None
        self._use_langchain = False

    @property
    def client(self) -> Any:
        """Get the Tavily client (lazy initialization).

        Returns:
            TavilyClient instance

        Raises:
            ValueError: If API key is not provided
        """
        if self._client is None:
            if not self.api_key:
                raise ValueError("Tavily API key is required for web search")

            # Try langchain_tavily first (newer package)
            try:
                from langchain_tavily import TavilySearch

                self._langchain_tool = TavilySearch(
                    max_results=5,
                    tavily_api_key=self.api_key,
                )
                self._use_langchain = True
                logger.info("Using langchain_tavily.TavilySearch")
            except ImportError:
                pass

            # Fallback to direct Tavily client
            try:
                from tavily import TavilyClient

                self._client = TavilyClient(api_key=self.api_key)
                logger.info("Using Tavily Python client directly")
            except ImportError as e:
                raise ImportError(
                    "Neither langchain_community.tools.TavilySearchResults nor "
                    "tavily-python is available. Install one of them."
                ) from e

        return self._client

    def search(
        self,
        query: str,
        max_results: int = 5,
        search_depth: str = "basic",
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
    ) -> list[SearchResult]:
        """Perform a web search and return structured results.

        Args:
            query: Search query
            max_results: Maximum number of results (default: 5)
            search_depth: "basic" or "advanced" (default: "basic")
            include_domains: Domains to include (optional)
            exclude_domains: Domains to exclude (optional)

        Returns:
            List of SearchResult objects with title, url, snippet, and relevance_score
        """
        # Check if API key is available
        if not self.api_key:
            logger.warning("No Tavily API key provided, returning mock results")
            return _get_mock_results(query)[:max_results]

        try:
            # Try using langchain tool if available
            if self._use_langchain and self._langchain_tool:
                return self._search_with_langchain(query, max_results)

            # Initialize client if needed
            _ = self.client

            # Use direct Tavily client
            response = self._client.search(
                query=query,
                max_results=max_results,
                search_depth=search_depth,
                include_domains=include_domains or [],
                exclude_domains=exclude_domains or [],
            )

            return self._parse_tavily_response(response)

        except Exception as e:
            logger.error(f"Tavily search failed: {e}. Falling back to mock results.")
            return _get_mock_results(query)[:max_results]

    def _search_with_langchain(
        self,
        query: str,
        max_results: int = 5,
    ) -> list[SearchResult]:
        """Search using langchain_community TavilySearchResults.

        Args:
            query: Search query
            max_results: Maximum number of results

        Returns:
            List of SearchResult objects
        """
        try:
            # Update max_results if different
            self._langchain_tool.max_results = max_results

            # Invoke the tool
            results = self._langchain_tool.invoke({"query": query})

            # Parse results (langchain returns list of dicts)
            search_results = []
            for i, result in enumerate(results):
                if isinstance(result, dict):
                    # Calculate a simple relevance score based on position
                    relevance = 1.0 - (i * 0.1) if i < 10 else 0.1

                    search_results.append(
                        SearchResult(
                            title=result.get("title", ""),
                            url=result.get("url", ""),
                            snippet=result.get("content", result.get("snippet", "")),
                            relevance_score=round(relevance, 2),
                        )
                    )

            return search_results

        except Exception as e:
            logger.warning(f"Langchain Tavily search failed: {e}")
            raise

    def _parse_tavily_response(self, response: dict) -> list[SearchResult]:
        """Parse Tavily API response into SearchResult objects.

        Args:
            response: Raw Tavily API response

        Returns:
            List of SearchResult objects
        """
        results = response.get("results", [])
        search_results = []

        for i, result in enumerate(results):
            # Tavily provides a score, normalize to 0-1 range
            raw_score = result.get("score", 0.0)
            # Scores are typically 0-1 already, but ensure bounds
            relevance = min(max(float(raw_score), 0.0), 1.0)

            search_results.append(
                SearchResult(
                    title=result.get("title", ""),
                    url=result.get("url", ""),
                    snippet=result.get("content", ""),
                    relevance_score=round(relevance, 3),
                )
            )

        return search_results

    def format_results_for_llm(
        self,
        results: list[SearchResult],
        include_scores: bool = False,
    ) -> str:
        """Format search results into a string suitable for LLM context.

        Args:
            results: List of SearchResult objects
            include_scores: Whether to include relevance scores in output

        Returns:
            Formatted string for LLM context
        """
        if not results:
            return "No web search results found."

        context_parts = ["## Web Search Results\n"]

        for i, result in enumerate(results, 1):
            header = f"### [{i}] {result.title}"
            if include_scores:
                header += f" (relevance: {result.relevance_score:.2f})"

            parts = [
                header,
                f"**URL:** {result.url}",
                f"{result.snippet}",
            ]
            context_parts.append("\n".join(parts))

        return "\n\n".join(context_parts)

    def get_search_context(
        self,
        query: str,
        max_results: int = 5,
    ) -> str:
        """Convenience method: Search and return formatted context for LLM.

        Args:
            query: Search query
            max_results: Maximum number of results

        Returns:
            Formatted context string ready for LLM consumption
        """
        results = self.search(query, max_results=max_results)
        return self.format_results_for_llm(results)

    def search_with_answer(self, query: str) -> dict:
        """Perform an advanced search and get a direct answer.

        Args:
            query: Search query

        Returns:
            Dictionary with 'answer' and 'results' keys
        """
        if not self.api_key:
            logger.warning("No Tavily API key provided, returning mock answer")
            return {
                "answer": f"Mock answer for: {query}",
                "results": _get_mock_results(query),
            }

        try:
            _ = self.client
            response = self._client.search(
                query=query,
                search_depth="advanced",
                include_answer=True,
            )
            return {
                "answer": response.get("answer", ""),
                "results": self._parse_tavily_response(response),
            }
        except Exception as e:
            logger.error(f"Tavily search with answer failed: {e}")
            return {
                "answer": "",
                "results": _get_mock_results(query),
            }


# --- Backward Compatibility Alias ---

# Alias for backward compatibility with existing code
WebSearchService = TavilySearchService
