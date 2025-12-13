"""Web Agent for answering questions using web search results.

This module provides a LangGraph node that searches the web and
synthesizes answers from the search results with source citations.
"""

from typing import TypedDict

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from app.core.config import settings
from app.services.web_search import SearchResult, TavilySearchService


class WebSource(BaseModel):
    """Source citation for a web search result."""

    title: str = Field(description="Page title")
    url: str = Field(description="Page URL")
    snippet: str = Field(default="", description="Content snippet from the source")
    relevance_score: float = Field(default=0.0, description="Relevance score (0.0-1.0)")


class WebAgentResult(BaseModel):
    """Result from the web agent."""

    answer: str = Field(description="The synthesized answer from web search")
    sources: list[WebSource] = Field(default_factory=list, description="Source URLs")
    found_results: bool = Field(
        default=True, description="Whether relevant web results were found"
    )


class WebAgentState(TypedDict):
    """State for the web agent."""

    query: str
    chat_history: list[BaseMessage]
    search_results: list[SearchResult]
    formatted_results: str
    answer: str
    sources: list[WebSource]
    found_results: bool


WEB_ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful research assistant that synthesizes information from web search results.

Your task is to:
1. Carefully read the web search results provided
2. Synthesize a comprehensive, accurate answer to the user's question
3. Cite your sources with URLs when making specific claims
4. Be clear about what information comes from which source

IMPORTANT RULES:
- Base your answer ONLY on the provided search results
- Always cite sources with their URLs
- If the search results don't contain relevant information, say so clearly
- Synthesize information from multiple sources when possible
- Be concise but thorough"""
    ),
    (
        "human",
        """Based on these web search results, answer the user's question.

Search Results:
{results}

Question: {question}

Provide a helpful answer and cite your sources with URLs."""
    ),
])


class WebAgent:
    """Agent for answering questions using web search results."""

    def __init__(
        self,
        model_name: str | None = None,
        max_search_results: int = 5,
    ):
        """Initialize the web agent.

        Args:
            model_name: LLM model name, defaults to settings.default_llm_model
            max_search_results: Maximum number of web search results to retrieve
        """
        self.model_name = model_name or settings.default_llm_model
        self.max_search_results = max_search_results

        self.llm = ChatOpenAI(
            model=self.model_name,
            temperature=0,
            api_key=settings.openai_api_key,
            base_url=settings.openai_api_base,
        )

        self.search_service = TavilySearchService(api_key=settings.tavily_api_key)

        # Build the graph
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(WebAgentState)

        # Add nodes
        workflow.add_node("search_web", self._search_web)
        workflow.add_node("check_results", self._check_results)
        workflow.add_node("synthesize_answer", self._synthesize_answer)
        workflow.add_node("handle_no_results", self._handle_no_results)

        # Add edges
        workflow.add_edge(START, "search_web")
        workflow.add_edge("search_web", "check_results")
        workflow.add_conditional_edges(
            "check_results",
            self._should_synthesize,
            {
                "synthesize": "synthesize_answer",
                "no_results": "handle_no_results",
            },
        )
        workflow.add_edge("synthesize_answer", END)
        workflow.add_edge("handle_no_results", END)

        return workflow.compile()

    def _search_web(self, state: WebAgentState) -> dict:
        """Perform web search.

        Args:
            state: Current agent state

        Returns:
            Updated state with search results
        """
        query = state["query"]

        # Perform web search
        search_results = self.search_service.search(
            query=query,
            max_results=self.max_search_results,
        )

        return {"search_results": search_results}

    def _check_results(self, state: WebAgentState) -> dict:
        """Check if web search returned relevant results.

        Args:
            state: Current agent state

        Returns:
            Updated state with formatted results and sources
        """
        search_results = state.get("search_results", [])

        if not search_results:
            return {
                "found_results": False,
                "sources": [],
                "formatted_results": "",
            }

        # Check if any results have meaningful content
        has_relevant = any(
            result.snippet and len(result.snippet) > 20
            for result in search_results
        )

        # Format results for LLM
        formatted_results = self.search_service.format_results_for_llm(
            search_results, include_scores=False
        )

        # Build source citations
        sources = [
            WebSource(
                title=result.title,
                url=result.url,
                snippet=result.snippet[:200] + "..." if len(result.snippet) > 200 else result.snippet,
                relevance_score=result.relevance_score,
            )
            for result in search_results
            if result.url  # Only include results with URLs
        ]

        return {
            "found_results": has_relevant,
            "sources": sources,
            "formatted_results": formatted_results,
        }

    def _should_synthesize(self, state: WebAgentState) -> str:
        """Determine if we should synthesize an answer or signal no results.

        Args:
            state: Current agent state

        Returns:
            Next node to execute
        """
        if state.get("found_results", False):
            return "synthesize"
        return "no_results"

    def _synthesize_answer(self, state: WebAgentState) -> dict:
        """Synthesize an answer from web search results.

        Args:
            state: Current agent state

        Returns:
            Updated state with synthesized answer
        """
        query = state["query"]
        formatted_results = state.get("formatted_results", "")

        chain = WEB_ANSWER_PROMPT | self.llm

        response = chain.invoke({
            "results": formatted_results,
            "question": query,
        })

        return {"answer": response.content}

    def _handle_no_results(self, state: WebAgentState) -> dict:
        """Handle case where no relevant web results were found.

        Args:
            state: Current agent state

        Returns:
            Updated state with no-results message
        """
        return {
            "answer": "I couldn't find relevant information from web search for your query. "
                      "The search may not have returned results, or the query might need "
                      "to be rephrased. Please try a different search term or ask "
                      "a more specific question.",
            "found_results": False,
        }

    async def arun(
        self,
        query: str,
        chat_history: list[BaseMessage] | None = None,
    ) -> WebAgentResult:
        """Run the web agent asynchronously.

        Args:
            query: User query to answer
            chat_history: Previous conversation messages

        Returns:
            WebAgentResult with answer and sources
        """
        initial_state: WebAgentState = {
            "query": query,
            "chat_history": chat_history or [],
            "search_results": [],
            "formatted_results": "",
            "answer": "",
            "sources": [],
            "found_results": False,
        }

        result = await self.graph.ainvoke(initial_state)

        return WebAgentResult(
            answer=result.get("answer", ""),
            sources=result.get("sources", []),
            found_results=result.get("found_results", False),
        )

    def run(
        self,
        query: str,
        chat_history: list[BaseMessage] | None = None,
    ) -> WebAgentResult:
        """Run the web agent synchronously.

        Args:
            query: User query to answer
            chat_history: Previous conversation messages

        Returns:
            WebAgentResult with answer and sources
        """
        initial_state: WebAgentState = {
            "query": query,
            "chat_history": chat_history or [],
            "search_results": [],
            "formatted_results": "",
            "answer": "",
            "sources": [],
            "found_results": False,
        }

        result = self.graph.invoke(initial_state)

        return WebAgentResult(
            answer=result.get("answer", ""),
            sources=result.get("sources", []),
            found_results=result.get("found_results", False),
        )


def create_web_agent(
    model_name: str | None = None,
    max_search_results: int = 5,
) -> WebAgent:
    """Create a web agent instance.

    Args:
        model_name: Optional LLM model name override
        max_search_results: Maximum number of search results to retrieve

    Returns:
        Configured WebAgent
    """
    return WebAgent(
        model_name=model_name,
        max_search_results=max_search_results,
    )


# Convenience function for use as a LangGraph node
def web_search(
    query: str,
    chat_history: list[BaseMessage] | None = None,
    model_name: str | None = None,
    max_search_results: int = 5,
) -> WebAgentResult:
    """Search the web and synthesize an answer.

    This is a convenience function that can be used directly without
    instantiating the full agent.

    Args:
        query: User query to answer
        chat_history: Previous conversation messages
        model_name: Optional LLM model name override
        max_search_results: Maximum number of search results

    Returns:
        WebAgentResult with answer and sources
    """
    agent = create_web_agent(
        model_name=model_name,
        max_search_results=max_search_results,
    )
    return agent.run(query=query, chat_history=chat_history)


# LangGraph node function for integration into larger graphs
def web_agent_node(state: dict) -> dict:
    """LangGraph node function for web search question answering.

    This function can be used directly as a node in a LangGraph workflow.
    Expects state to have 'query' and optionally 'chat_history' keys.

    Args:
        state: Graph state containing query and chat_history

    Returns:
        Updated state with answer, sources, and found_results
    """
    query = state.get("query", "")
    chat_history = state.get("chat_history", [])

    result = web_search(query=query, chat_history=chat_history)

    return {
        "answer": result.answer,
        "sources": [source.model_dump() for source in result.sources],
        "found_results": result.found_results,
    }
