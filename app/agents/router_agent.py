"""Router Agent for deciding which tool/agent to use based on the query.

This module provides a LangGraph node that autonomously decides the best
approach for answering a user's question using LLM reasoning.
"""

import json
from typing import Literal, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from app.core.config import settings


# Route types
RouteType = Literal["pdf_search", "web_search", "hybrid", "clarify"]


class RouterResult(BaseModel):
    """Result of routing decision."""

    route: RouteType = Field(
        description="The selected route: pdf_search, web_search, hybrid, or clarify"
    )
    reasoning: str = Field(
        description="Step-by-step reasoning for the routing decision"
    )
    sub_queries: list[str] = Field(
        default_factory=list,
        description="Sub-queries if the question needs to be broken down"
    )
    original_query: str = Field(description="The original user query")


class RouterState(TypedDict):
    """State for the router agent."""

    query: str
    chat_history: list[BaseMessage]
    route: RouteType | None
    reasoning: str | None
    sub_queries: list[str]


ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an expert query router for a RAG (Retrieval Augmented Generation) system.
Your task is to analyze user questions and decide the best approach for answering them.

You have access to the following tools:
1. **pdf_search** - Search through academic papers on text-to-SQL, prompt engineering, and related AI/ML topics
2. **web_search** - Search the current internet for up-to-date information, news, or external resources

You must decide one of these routes:
- **pdf_search**: Use when the query can be answered from academic papers (text-to-SQL techniques, prompt engineering methods, research findings, theoretical concepts)
- **web_search**: Use when the query requires current/external information (recent news, live data, external websites, documentation for specific tools)
- **hybrid**: Use when you need BOTH PDF context AND web search (e.g., comparing research findings with current implementations, or supplementing academic knowledge with recent developments)
- **clarify**: Use when the question is too vague, ambiguous, or lacks necessary context to route properly

Think step-by-step about:
1. What kind of information does this question need?
2. Would academic papers on text-to-SQL or prompt engineering contain the answer?
3. Does this require current/real-time information?
4. Is the question clear enough to answer, or too vague?
5. Can this question be broken into smaller, more focused sub-queries?

For complex questions that span multiple topics, break them into sub-queries. Each sub-query should be:
- Self-contained and answerable independently
- Focused on a single aspect of the original question
- Useful for retrieving relevant information

Respond with a JSON object in this exact format:
{{
    "route": "pdf_search" | "web_search" | "hybrid" | "clarify",
    "reasoning": "Your step-by-step reasoning explaining why you chose this route",
    "sub_queries": ["list", "of", "sub-queries"] // empty list if no breakdown needed
}}"""
    ),
    (
        "human",
        """Given this question and available tools, decide the best approach.

Question: {question}

Chat History:
{history}

Available tools:
- pdf_search: Academic papers on text-to-SQL, prompt engineering
- web_search: Current internet information

Think step-by-step about what information sources are needed.
Respond in JSON only, no other text."""
    ),
])


class RouterAgent:
    """Agent for routing queries to the appropriate tool/agent."""

    def __init__(self, model_name: str | None = None):
        """Initialize the router agent.

        Args:
            model_name: LLM model name, defaults to settings.default_llm_model
        """
        self.model_name = model_name or settings.default_llm_model
        self.llm = ChatOpenAI(
            model=self.model_name,
            temperature=0,
            api_key=settings.openai_api_key,
            base_url=settings.openai_api_base,
        )

        # Build the graph
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(RouterState)

        # Add nodes
        workflow.add_node("route", self._route_query)

        # Add edges
        workflow.add_edge(START, "route")
        workflow.add_edge("route", END)

        return workflow.compile()

    def _format_history(self, messages: list[BaseMessage]) -> str:
        """Format chat history for the prompt.

        Args:
            messages: List of chat messages

        Returns:
            Formatted string representation of chat history
        """
        if not messages:
            return "(No previous conversation)"

        formatted_parts = []
        for msg in messages[-10:]:  # Last 10 messages for context
            if isinstance(msg, HumanMessage):
                role = "User"
            elif isinstance(msg, AIMessage):
                role = "Assistant"
            else:
                role = "System"
            formatted_parts.append(f"{role}: {msg.content}")

        return "\n".join(formatted_parts)

    def _parse_route(self, route_str: str) -> RouteType:
        """Parse and validate the route string.

        Args:
            route_str: Route string from LLM response

        Returns:
            Valid RouteType
        """
        route_str = route_str.lower().strip()
        valid_routes: list[RouteType] = ["pdf_search", "web_search", "hybrid", "clarify"]

        if route_str in valid_routes:
            return route_str  # type: ignore

        # Try to match partial routes
        for valid_route in valid_routes:
            if route_str in valid_route or valid_route in route_str:
                return valid_route

        # Default to clarify if unable to parse
        return "clarify"

    def _route_query(self, state: RouterState) -> dict:
        """Route the query using LLM reasoning.

        Args:
            state: Current agent state

        Returns:
            Updated state with routing decision
        """
        query = state["query"]
        history = self._format_history(state.get("chat_history", []))

        # Generate routing decision using LLM
        chain = ROUTER_PROMPT | self.llm

        try:
            response = chain.invoke({
                "question": query,
                "history": history,
            })

            # Parse JSON response
            content = response.content.strip()
            # Handle markdown code blocks if present
            if content.startswith("```"):
                content = content.split("\n", 1)[1]
                if content.endswith("```"):
                    content = content.rsplit("```", 1)[0]
                content = content.strip()

            result = json.loads(content)

            route = self._parse_route(result.get("route", "clarify"))
            sub_queries = result.get("sub_queries", [])

            # Ensure sub_queries is a list of strings
            if not isinstance(sub_queries, list):
                sub_queries = []
            sub_queries = [str(sq) for sq in sub_queries if sq]

            return {
                "route": route,
                "reasoning": result.get("reasoning", "No reasoning provided"),
                "sub_queries": sub_queries,
            }
        except (json.JSONDecodeError, KeyError) as e:
            # On parsing error, default to clarify with error info
            return {
                "route": "clarify",
                "reasoning": f"Routing analysis failed: {e!s}. Defaulting to clarification.",
                "sub_queries": [],
            }

    async def arun(
        self,
        query: str,
        chat_history: list[BaseMessage] | None = None,
    ) -> RouterResult:
        """Run the router agent asynchronously.

        Args:
            query: User query to route
            chat_history: Previous conversation messages

        Returns:
            RouterResult with routing decision
        """
        initial_state: RouterState = {
            "query": query,
            "chat_history": chat_history or [],
            "route": None,
            "reasoning": None,
            "sub_queries": [],
        }

        # Run the graph
        result = await self.graph.ainvoke(initial_state)

        return RouterResult(
            route=result["route"],
            reasoning=result["reasoning"] or "",
            sub_queries=result.get("sub_queries", []),
            original_query=query,
        )

    def run(
        self,
        query: str,
        chat_history: list[BaseMessage] | None = None,
    ) -> RouterResult:
        """Run the router agent synchronously.

        Args:
            query: User query to route
            chat_history: Previous conversation messages

        Returns:
            RouterResult with routing decision
        """
        initial_state: RouterState = {
            "query": query,
            "chat_history": chat_history or [],
            "route": None,
            "reasoning": None,
            "sub_queries": [],
        }

        # Run the graph
        result = self.graph.invoke(initial_state)

        return RouterResult(
            route=result["route"],
            reasoning=result["reasoning"] or "",
            sub_queries=result.get("sub_queries", []),
            original_query=query,
        )


def create_router_agent(
    model_name: str | None = None,
) -> RouterAgent:
    """Create a router agent instance.

    Args:
        model_name: Optional LLM model name override

    Returns:
        Configured RouterAgent
    """
    return RouterAgent(model_name=model_name)


# Convenience function for use as a LangGraph node
def route_query(
    query: str,
    chat_history: list[BaseMessage] | None = None,
    model_name: str | None = None,
) -> RouterResult:
    """Route a query to the appropriate tool/agent.

    This is a convenience function that can be used directly without
    instantiating the full agent.

    Args:
        query: User query to route
        chat_history: Previous conversation messages
        model_name: Optional LLM model name override

    Returns:
        RouterResult with routing decision
    """
    agent = create_router_agent(model_name=model_name)
    return agent.run(query=query, chat_history=chat_history)


# LangGraph node function for integration into larger graphs
def router_node(state: dict) -> dict:
    """LangGraph node function for routing queries.

    This function can be used directly as a node in a LangGraph workflow.
    Expects state to have 'query' and optionally 'chat_history' keys.

    Args:
        state: Graph state containing query and chat_history

    Returns:
        Updated state with routing decision
    """
    query = state.get("query", "")
    chat_history = state.get("chat_history", [])

    result = route_query(query=query, chat_history=chat_history)

    return {
        "route": result.route,
        "reasoning": result.reasoning,
        "sub_queries": result.sub_queries,
    }
