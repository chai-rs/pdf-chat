"""Clarification Agent for detecting ambiguous queries.

This module provides a LangGraph node that analyzes user questions for ambiguity
and determines if clarification is needed before processing.
"""

import json
from typing import TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from app.core.config import settings


class ClarificationResult(BaseModel):
    """Result of clarification analysis."""

    needs_clarification: bool = Field(description="Whether the query needs clarification")
    clarification_question: str | None = Field(
        default=None, description="Suggested clarification question if needed"
    )
    original_query: str = Field(description="The original user query")
    reason: str | None = Field(default=None, description="Reason why clarification is needed")


class ClarificationState(TypedDict):
    """State for the clarification agent."""

    query: str
    chat_history: list[BaseMessage]
    needs_clarification: bool
    clarification_question: str | None
    reason: str | None


CLARIFICATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert at analyzing questions for ambiguity and clarity.
Your task is to determine if a user's question is clear enough to answer accurately,
or if it requires clarification first.

Consider the following when analyzing:
1. Pronouns like "it", "this", "that", "they" - do they have clear referents in the chat history?
2. Vague terms like "enough", "good", "better", "more", "less" - are they defined with specific criteria?
3. Ambiguous scope - is it clear what exactly the user is asking about?
4. Missing context - does the question assume knowledge that isn't provided?
5. Multiple interpretations - could the question reasonably be interpreted in different ways?

IMPORTANT: Use the chat history to resolve ambiguities. If pronouns or references are clear
from the conversation context, the question is NOT ambiguous.

Respond with a JSON object in this exact format:
{{
    "needs_clarification": true/false,
    "reason": "Brief explanation of why clarification is needed (or null if not needed)",
    "suggested_question": "A polite clarification question to ask the user (or null if not needed)"
}}

Only mark as needing clarification if the ambiguity would significantly impact the quality
of the response. Minor ambiguities that don't affect the core question should be ignored.""",
        ),
        (
            "human",
            """Analyze if this question needs clarification.

Chat History:
{history}

Current Question: {question}

Respond with JSON only, no other text.""",
        ),
    ]
)


class ClarificationAgent:
    """Agent for detecting ambiguous queries that need clarification."""

    def __init__(self, model_name: str | None = None):
        """Initialize the clarification agent.

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
        workflow = StateGraph(ClarificationState)

        # Add nodes
        workflow.add_node("analyze", self._analyze_query)

        # Add edges
        workflow.add_edge(START, "analyze")
        workflow.add_edge("analyze", END)

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

    def _analyze_query(self, state: ClarificationState) -> dict:
        """Analyze the query for ambiguity using LLM.

        Args:
            state: Current agent state

        Returns:
            Updated state with clarification analysis results
        """
        query = state["query"]
        history = self._format_history(state.get("chat_history", []))

        # Generate analysis using LLM
        chain = CLARIFICATION_PROMPT | self.llm

        try:
            response = chain.invoke(
                {
                    "question": query,
                    "history": history,
                }
            )

            # Parse JSON response
            content = response.content.strip()
            # Handle markdown code blocks if present
            if content.startswith("```"):
                content = content.split("\n", 1)[1]
                if content.endswith("```"):
                    content = content.rsplit("```", 1)[0]
                content = content.strip()

            result = json.loads(content)

            return {
                "needs_clarification": result.get("needs_clarification", False),
                "clarification_question": result.get("suggested_question"),
                "reason": result.get("reason"),
            }
        except (json.JSONDecodeError, KeyError) as e:
            # On parsing error, assume no clarification needed
            return {
                "needs_clarification": False,
                "clarification_question": None,
                "reason": f"Analysis failed: {e!s}",
            }

    async def arun(
        self,
        query: str,
        chat_history: list[BaseMessage] | None = None,
    ) -> ClarificationResult:
        """Run the clarification agent asynchronously.

        Args:
            query: User query to analyze
            chat_history: Previous conversation messages

        Returns:
            ClarificationResult with analysis
        """
        initial_state: ClarificationState = {
            "query": query,
            "chat_history": chat_history or [],
            "needs_clarification": False,
            "clarification_question": None,
            "reason": None,
        }

        # Run the graph
        result = await self.graph.ainvoke(initial_state)

        return ClarificationResult(
            needs_clarification=result["needs_clarification"],
            clarification_question=result.get("clarification_question"),
            original_query=query,
            reason=result.get("reason"),
        )

    def run(
        self,
        query: str,
        chat_history: list[BaseMessage] | None = None,
    ) -> ClarificationResult:
        """Run the clarification agent synchronously.

        Args:
            query: User query to analyze
            chat_history: Previous conversation messages

        Returns:
            ClarificationResult with analysis
        """
        initial_state: ClarificationState = {
            "query": query,
            "chat_history": chat_history or [],
            "needs_clarification": False,
            "clarification_question": None,
            "reason": None,
        }

        # Run the graph
        result = self.graph.invoke(initial_state)

        return ClarificationResult(
            needs_clarification=result["needs_clarification"],
            clarification_question=result.get("clarification_question"),
            original_query=query,
            reason=result.get("reason"),
        )


def create_clarification_agent(
    model_name: str | None = None,
) -> ClarificationAgent:
    """Create a clarification agent instance.

    Args:
        model_name: Optional LLM model name override

    Returns:
        Configured ClarificationAgent
    """
    return ClarificationAgent(model_name=model_name)


# Convenience function for use as a LangGraph node
def check_clarification_needed(
    query: str,
    chat_history: list[BaseMessage] | None = None,
    model_name: str | None = None,
) -> ClarificationResult:
    """Check if a query needs clarification.

    This is a convenience function that can be used directly without
    instantiating the full agent.

    Args:
        query: User query to analyze
        chat_history: Previous conversation messages
        model_name: Optional LLM model name override

    Returns:
        ClarificationResult with analysis
    """
    agent = create_clarification_agent(model_name=model_name)
    return agent.run(query=query, chat_history=chat_history)
