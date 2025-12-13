"""Orchestrator Agent - Main LangGraph workflow connecting all agents.

This module provides the central workflow that coordinates:
- Clarification checking
- Query routing
- PDF and web search execution
- Answer synthesis
"""

import logging
from typing import Literal, TypedDict

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.agents.clarification_agent import ClarificationAgent
from app.agents.pdf_agent import PDFAgent
from app.agents.router_agent import RouterAgent
from app.agents.web_agent import WebAgent
from app.core.config import settings


logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """State schema for the orchestrator workflow."""

    question: str
    session_id: str
    chat_history: list[BaseMessage]
    route: str | None
    sub_queries: list[str]
    pdf_results: list[dict]
    web_results: list[dict]
    final_answer: str
    sources: list[dict]
    needs_clarification: bool
    clarification_question: str | None
    # Internal state for re-planning
    retrieval_failed: bool
    attempt_count: int


# Conditional edge types
CLARIFY_RESPONSE = "clarify_response"
ROUTER = "router"
PDF_AGENT = "pdf_agent"
WEB_AGENT = "web_agent"
SYNTHESIZER = "synthesizer"
REPLAN = "replan"


SYNTHESIS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful research assistant that synthesizes information from multiple sources.

Your task is to:
1. Combine information from PDF documents and web search results
2. Create a comprehensive, well-organized answer
3. Cite sources appropriately (mention document names for PDFs, URLs for web)
4. Resolve any contradictions between sources by preferring more authoritative/recent information
5. Be clear about the origin of each piece of information

Guidelines:
- If sources disagree, acknowledge both perspectives
- Prioritize accuracy over comprehensiveness
- Use a clear structure (introduction, main points, conclusion if appropriate)
- Include relevant citations inline""",
        ),
        (
            "human",
            """Based on the following research results, provide a comprehensive answer to the user's question.

Question: {question}

PDF Document Results:
{pdf_results}

Web Search Results:
{web_results}

Synthesize this information into a clear, well-organized answer with proper citations.""",
        ),
    ]
)


class Orchestrator:
    """Main orchestrator that coordinates all agents in the RAG pipeline."""

    def __init__(
        self,
        model_name: str | None = None,
        max_attempts: int = 2,
    ):
        """Initialize the orchestrator.

        Args:
            model_name: LLM model name, defaults to settings.default_llm_model
            max_attempts: Maximum attempts for re-planning if retrieval fails
        """
        self.model_name = model_name or settings.default_llm_model
        self.max_attempts = max_attempts

        # Initialize sub-agents
        self.clarification_agent = ClarificationAgent(model_name=self.model_name)
        self.router_agent = RouterAgent(model_name=self.model_name)
        self.pdf_agent = PDFAgent(model_name=self.model_name)
        self.web_agent = WebAgent(model_name=self.model_name)

        # LLM for synthesis
        self.llm = ChatOpenAI(
            model=self.model_name,
            temperature=0.3,
            api_key=settings.openai_api_key,
            base_url=settings.openai_api_base,
        )

        # Build the graph with checkpointing
        self.checkpointer = MemorySaver()
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the main LangGraph workflow."""
        workflow = StateGraph(AgentState)

        # Add all nodes
        workflow.add_node("clarification_check", self._clarification_check)
        workflow.add_node("router", self._route_query)
        workflow.add_node("pdf_agent", self._run_pdf_agent)
        workflow.add_node("web_agent", self._run_web_agent)
        workflow.add_node("synthesizer", self._synthesize_answer)
        workflow.add_node("replan", self._replan)

        # START -> clarification_check
        workflow.add_edge(START, "clarification_check")

        # Conditional edge: clarification_check -> (END with clarification | router)
        workflow.add_conditional_edges(
            "clarification_check",
            self._after_clarification_check,
            {
                CLARIFY_RESPONSE: END,
                ROUTER: "router",
            },
        )

        # Conditional edge: router -> (pdf_agent | web_agent | BOTH based on route)
        workflow.add_conditional_edges(
            "router",
            self._after_router,
            {
                PDF_AGENT: "pdf_agent",
                WEB_AGENT: "web_agent",
                CLARIFY_RESPONSE: END,
            },
        )

        # Conditional edge: pdf_agent -> (web_agent for hybrid | synthesizer | replan)
        workflow.add_conditional_edges(
            "pdf_agent",
            self._after_pdf_agent,
            {
                WEB_AGENT: "web_agent",
                SYNTHESIZER: "synthesizer",
                REPLAN: "replan",
            },
        )

        # Conditional edge: web_agent -> (synthesizer | replan)
        workflow.add_conditional_edges(
            "web_agent",
            self._after_web_agent,
            {
                SYNTHESIZER: "synthesizer",
                REPLAN: "replan",
            },
        )

        # replan -> router (to try alternative route)
        workflow.add_edge("replan", "router")

        # synthesizer -> END
        workflow.add_edge("synthesizer", END)

        return workflow.compile(checkpointer=self.checkpointer)

    # -------------------------------------------------------------------------
    # Node implementations
    # -------------------------------------------------------------------------

    def _clarification_check(self, state: AgentState) -> dict:
        """Check if the query needs clarification.

        Args:
            state: Current agent state

        Returns:
            Updated state with clarification status
        """
        result = self.clarification_agent.run(
            query=state["question"],
            chat_history=state.get("chat_history", []),
        )

        return {
            "needs_clarification": result.needs_clarification,
            "clarification_question": result.clarification_question,
        }

    def _route_query(self, state: AgentState) -> dict:
        """Route the query to appropriate agent(s).

        Args:
            state: Current agent state

        Returns:
            Updated state with routing decision
        """
        result = self.router_agent.run(
            query=state["question"],
            chat_history=state.get("chat_history", []),
        )

        return {
            "route": result.route,
            "sub_queries": result.sub_queries,
        }

    def _run_pdf_agent(self, state: AgentState) -> dict:
        """Run the PDF search agent.

        Args:
            state: Current agent state

        Returns:
            Updated state with PDF search results
        """
        # Use sub-queries if available, otherwise use main question
        queries = state.get("sub_queries", []) or [state["question"]]

        all_results = []
        all_sources = []

        for query in queries:
            result = self.pdf_agent.run(
                query=query,
                chat_history=state.get("chat_history", []),
            )

            all_results.append(
                {
                    "query": query,
                    "answer": result.answer,
                    "found_in_pdfs": result.found_in_pdfs,
                    "confidence_score": result.confidence_score,
                }
            )

            # Convert Source objects to dicts
            for source in result.sources:
                all_sources.append(
                    {
                        "type": "pdf",
                        "filename": source.filename,
                        "page": source.page,
                        "snippet": source.snippet,
                        "relevance_score": source.relevance_score,
                    }
                )

        # Check if retrieval was successful
        retrieval_failed = all(not r.get("found_in_pdfs", False) for r in all_results)

        return {
            "pdf_results": all_results,
            "sources": state.get("sources", []) + all_sources,
            "retrieval_failed": retrieval_failed,
        }

    def _run_web_agent(self, state: AgentState) -> dict:
        """Run the web search agent.

        Args:
            state: Current agent state

        Returns:
            Updated state with web search results
        """
        # Use sub-queries if available, otherwise use main question
        queries = state.get("sub_queries", []) or [state["question"]]

        all_results = []
        all_sources = []

        for query in queries:
            result = self.web_agent.run(
                query=query,
                chat_history=state.get("chat_history", []),
            )

            all_results.append(
                {
                    "query": query,
                    "answer": result.answer,
                    "found_results": result.found_results,
                }
            )

            # Convert WebSource objects to dicts
            for source in result.sources:
                all_sources.append(
                    {
                        "type": "web",
                        "title": source.title,
                        "url": source.url,
                        "snippet": source.snippet,
                        "relevance_score": source.relevance_score,
                    }
                )

        # Check if retrieval was successful
        pdf_failed = state.get("retrieval_failed", False)
        web_failed = all(not r.get("found_results", False) for r in all_results)

        # Both failed = overall retrieval failed
        retrieval_failed = (
            pdf_failed and web_failed if state.get("route") == "hybrid" else web_failed
        )

        return {
            "web_results": all_results,
            "sources": state.get("sources", []) + all_sources,
            "retrieval_failed": retrieval_failed,
        }

    def _synthesize_answer(self, state: AgentState) -> dict:
        """Synthesize final answer from all results.

        Args:
            state: Current agent state

        Returns:
            Updated state with final answer
        """
        route = state.get("route", "hybrid")
        pdf_results = state.get("pdf_results", [])
        web_results = state.get("web_results", [])

        # For single-source routes, use the answer directly
        if route == "pdf_search" and pdf_results:
            # Combine PDF answers
            answers = [r.get("answer", "") for r in pdf_results if r.get("answer")]
            if len(answers) == 1:
                return {"final_answer": answers[0]}
            final_answer = "\n\n".join(answers)
            return {"final_answer": final_answer}

        if route == "web_search" and web_results:
            # Combine web answers
            answers = [r.get("answer", "") for r in web_results if r.get("answer")]
            if len(answers) == 1:
                return {"final_answer": answers[0]}
            final_answer = "\n\n".join(answers)
            return {"final_answer": final_answer}

        # For hybrid, synthesize both sources
        pdf_text = self._format_results(pdf_results, "PDF")
        web_text = self._format_results(web_results, "Web")

        if not pdf_text and not web_text:
            return {
                "final_answer": "I couldn't find relevant information to answer your question. "
                "Please try rephrasing your question or providing more context."
            }

        chain = SYNTHESIS_PROMPT | self.llm

        try:
            response = chain.invoke(
                {
                    "question": state["question"],
                    "pdf_results": pdf_text or "(No PDF results)",
                    "web_results": web_text or "(No web results)",
                }
            )
            return {"final_answer": response.content}
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            # Fallback: combine available answers
            all_answers = []
            all_answers.extend(r.get("answer", "") for r in pdf_results if r.get("answer"))
            all_answers.extend(r.get("answer", "") for r in web_results if r.get("answer"))
            return {"final_answer": "\n\n".join(all_answers) or "Failed to synthesize an answer."}

    def _replan(self, state: AgentState) -> dict:
        """Re-plan the approach after retrieval failure.

        Args:
            state: Current agent state

        Returns:
            Updated state with new routing strategy
        """
        current_attempt = state.get("attempt_count", 0) + 1
        current_route = state.get("route", "pdf_search")

        # Switch to alternative route
        if current_route == "pdf_search":
            new_route = "web_search"
        elif current_route == "web_search":
            new_route = "pdf_search"
        else:
            # For hybrid that failed, we've exhausted options
            new_route = None

        logger.info(
            f"Re-planning: attempt {current_attempt}, switching from {current_route} to {new_route}"
        )

        return {
            "route": new_route,
            "attempt_count": current_attempt,
            "retrieval_failed": False,  # Reset for new attempt
        }

    # -------------------------------------------------------------------------
    # Conditional edge functions
    # -------------------------------------------------------------------------

    def _after_clarification_check(
        self, state: AgentState
    ) -> Literal["clarify_response", "router"]:
        """Determine next step after clarification check.

        Args:
            state: Current agent state

        Returns:
            Next node key
        """
        if state.get("needs_clarification", False):
            return CLARIFY_RESPONSE
        return ROUTER

    def _after_router(
        self, state: AgentState
    ) -> Literal["pdf_agent", "web_agent", "clarify_response"]:
        """Determine next step after routing.

        Args:
            state: Current agent state

        Returns:
            Next node key
        """
        route = state.get("route", "clarify")

        if route == "clarify":
            return CLARIFY_RESPONSE
        elif route == "pdf_search":
            return PDF_AGENT
        elif route == "web_search":
            return WEB_AGENT
        elif route == "hybrid":
            # For hybrid, start with PDF
            return PDF_AGENT
        else:
            # Default to clarification
            return CLARIFY_RESPONSE

    def _after_pdf_agent(self, state: AgentState) -> Literal["web_agent", "synthesizer", "replan"]:
        """Determine next step after PDF agent.

        Args:
            state: Current agent state

        Returns:
            Next node key
        """
        route = state.get("route", "pdf_search")
        retrieval_failed = state.get("retrieval_failed", False)
        attempt_count = state.get("attempt_count", 0)

        # For hybrid, always continue to web agent
        if route == "hybrid":
            return WEB_AGENT

        # Check if we should re-plan
        if retrieval_failed and attempt_count < self.max_attempts:
            return REPLAN

        # Go to synthesizer
        return SYNTHESIZER

    def _after_web_agent(self, state: AgentState) -> Literal["synthesizer", "replan"]:
        """Determine next step after web agent.

        Args:
            state: Current agent state

        Returns:
            Next node key
        """
        route = state.get("route", "web_search")
        retrieval_failed = state.get("retrieval_failed", False)
        attempt_count = state.get("attempt_count", 0)

        # For hybrid, if both failed, still go to synthesizer (it will handle gracefully)
        if route == "hybrid":
            return SYNTHESIZER

        # Check if we should re-plan
        if retrieval_failed and attempt_count < self.max_attempts:
            return REPLAN

        # Go to synthesizer
        return SYNTHESIZER

    # -------------------------------------------------------------------------
    # Helper methods
    # -------------------------------------------------------------------------

    def _format_results(self, results: list[dict], source_type: str) -> str:
        """Format results for synthesis prompt.

        Args:
            results: List of result dictionaries
            source_type: Type of source (PDF or Web)

        Returns:
            Formatted string
        """
        if not results:
            return ""

        formatted_parts = []
        for i, result in enumerate(results, 1):
            answer = result.get("answer", "")
            query = result.get("query", "")
            if answer:
                formatted_parts.append(
                    f"### {source_type} Result {i}\n**Query**: {query}\n**Answer**: {answer}\n"
                )

        return "\n".join(formatted_parts)

    # -------------------------------------------------------------------------
    # Public interface
    # -------------------------------------------------------------------------

    async def arun(
        self,
        question: str,
        session_id: str,
        chat_history: list[BaseMessage] | None = None,
    ) -> AgentState:
        """Run the orchestrator asynchronously.

        Args:
            question: User question
            session_id: Session identifier for checkpointing
            chat_history: Previous conversation messages

        Returns:
            Final agent state
        """
        initial_state: AgentState = {
            "question": question,
            "session_id": session_id,
            "chat_history": chat_history or [],
            "route": None,
            "sub_queries": [],
            "pdf_results": [],
            "web_results": [],
            "final_answer": "",
            "sources": [],
            "needs_clarification": False,
            "clarification_question": None,
            "retrieval_failed": False,
            "attempt_count": 0,
        }

        config = {"configurable": {"thread_id": session_id}}
        result = await self.graph.ainvoke(initial_state, config=config)

        return result

    def run(
        self,
        question: str,
        session_id: str,
        chat_history: list[BaseMessage] | None = None,
    ) -> AgentState:
        """Run the orchestrator synchronously.

        Args:
            question: User question
            session_id: Session identifier for checkpointing
            chat_history: Previous conversation messages

        Returns:
            Final agent state
        """
        initial_state: AgentState = {
            "question": question,
            "session_id": session_id,
            "chat_history": chat_history or [],
            "route": None,
            "sub_queries": [],
            "pdf_results": [],
            "web_results": [],
            "final_answer": "",
            "sources": [],
            "needs_clarification": False,
            "clarification_question": None,
            "retrieval_failed": False,
            "attempt_count": 0,
        }

        config = {"configurable": {"thread_id": session_id}}
        result = self.graph.invoke(initial_state, config=config)

        return result

    def get_graph_visualization(self) -> str:
        """Get a Mermaid diagram representation of the graph.

        Returns:
            Mermaid diagram string
        """
        try:
            return self.graph.get_graph().draw_mermaid()
        except Exception:
            return "Graph visualization not available"


def create_orchestrator(
    model_name: str | None = None,
    max_attempts: int = 2,
) -> Orchestrator:
    """Create an orchestrator instance.

    Args:
        model_name: Optional LLM model name override
        max_attempts: Maximum re-planning attempts

    Returns:
        Configured Orchestrator
    """
    return Orchestrator(model_name=model_name, max_attempts=max_attempts)


# Convenience function for quick usage
def process_question(
    question: str,
    session_id: str,
    chat_history: list[BaseMessage] | None = None,
    model_name: str | None = None,
) -> AgentState:
    """Process a question through the orchestrator.

    This is a convenience function for quick usage without
    manually instantiating the orchestrator.

    Args:
        question: User question
        session_id: Session identifier
        chat_history: Previous conversation messages
        model_name: Optional LLM model name override

    Returns:
        Final agent state with answer
    """
    orchestrator = create_orchestrator(model_name=model_name)
    return orchestrator.run(
        question=question,
        session_id=session_id,
        chat_history=chat_history,
    )
