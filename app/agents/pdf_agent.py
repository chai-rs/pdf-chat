"""PDF Agent for answering questions using retrieved PDF content.

This module provides a LangGraph node that retrieves relevant chunks from
a vector store and generates answers grounded in the retrieved context.
"""

from typing import TypedDict

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from app.core.config import settings
from app.db.vector_store import get_vector_store


class Source(BaseModel):
    """Source citation for an answer."""

    filename: str = Field(description="Name of the source PDF file")
    page: int = Field(description="Page number in the PDF")
    chunk_index: int = Field(default=0, description="Chunk index within the page")
    relevance_score: float = Field(description="Relevance score of the chunk")
    snippet: str = Field(default="", description="Short snippet from the source")


class PDFAgentResult(BaseModel):
    """Result from the PDF agent."""

    answer: str = Field(description="The generated answer")
    sources: list[Source] = Field(default_factory=list, description="Source citations")
    confidence_score: float = Field(
        description="Confidence score based on retrieval relevance (0.0 to 1.0)"
    )
    found_in_pdfs: bool = Field(
        default=True, description="Whether relevant information was found in PDFs"
    )


class PDFAgentState(TypedDict):
    """State for the PDF agent."""

    query: str
    chat_history: list[BaseMessage]
    retrieved_docs: list[Document]
    context: str
    answer: str
    sources: list[Source]
    confidence_score: float
    found_in_pdfs: bool


# Default relevance threshold for determining if content is found
DEFAULT_RELEVANCE_THRESHOLD = 0.7

PDF_ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert research assistant that answers questions based ONLY on the provided context from academic papers.

Your task is to:
1. Carefully read the context provided
2. Answer the question using ONLY information from the context
3. Cite specific papers when possible using the format [Source: filename, page X]
4. If the answer is not in the context, clearly state that

IMPORTANT RULES:
- Do NOT make up information or use external knowledge
- Do NOT hallucinate or invent citations
- Be precise and accurate in your citations
- If you're uncertain, express that uncertainty""",
        ),
        (
            "human",
            """Answer the question based ONLY on the following context from academic papers.
If the answer is not in the context, say 'I could not find this information in the provided documents.'

Context:
{context}

Question: {question}

Provide a detailed answer with citations to specific papers when possible.""",
        ),
    ]
)


class PDFAgent:
    """Agent for answering questions using retrieved PDF content."""

    def __init__(
        self,
        model_name: str | None = None,
        relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
        k: int = 5,
    ):
        """Initialize the PDF agent.

        Args:
            model_name: LLM model name, defaults to settings.default_llm_model
            relevance_threshold: Minimum relevance score to consider content found
            k: Number of documents to retrieve
        """
        self.model_name = model_name or settings.default_llm_model
        self.relevance_threshold = relevance_threshold
        self.k = k

        self.llm = ChatOpenAI(
            model=self.model_name,
            temperature=0,
            api_key=settings.openai_api_key,
            base_url=settings.openai_api_base,
        )

        self.vector_store = get_vector_store()

        # Build the graph
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(PDFAgentState)

        # Add nodes
        workflow.add_node("retrieve", self._retrieve_documents)
        workflow.add_node("check_relevance", self._check_relevance)
        workflow.add_node("generate_answer", self._generate_answer)
        workflow.add_node("handle_not_found", self._handle_not_found)

        # Add edges
        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "check_relevance")
        workflow.add_conditional_edges(
            "check_relevance",
            self._should_generate_answer,
            {
                "generate": "generate_answer",
                "not_found": "handle_not_found",
            },
        )
        workflow.add_edge("generate_answer", END)
        workflow.add_edge("handle_not_found", END)

        return workflow.compile()

    def _retrieve_documents(self, state: PDFAgentState) -> dict:
        """Retrieve relevant documents from the vector store.

        Args:
            state: Current agent state

        Returns:
            Updated state with retrieved documents
        """
        query = state["query"]

        # Search with relevance scores
        results = self.vector_store.similarity_search_with_score(query, k=self.k)

        # Process results and extract documents
        retrieved_docs = []
        for doc, score in results:
            # Add score to metadata for later use
            doc.metadata["relevance_score"] = score
            retrieved_docs.append(doc)

        return {"retrieved_docs": retrieved_docs}

    def _check_relevance(self, state: PDFAgentState) -> dict:
        """Check if retrieved documents are relevant enough.

        Args:
            state: Current agent state

        Returns:
            Updated state with relevance information
        """
        retrieved_docs = state.get("retrieved_docs", [])

        if not retrieved_docs:
            return {
                "found_in_pdfs": False,
                "confidence_score": 0.0,
                "sources": [],
                "context": "",
            }

        # Calculate average relevance score
        # Note: ChromaDB returns distance scores (lower = more similar)
        scores = [doc.metadata.get("relevance_score", 1.0) for doc in retrieved_docs]

        # Check if any document meets the threshold
        # Lower distance = higher relevance
        has_relevant = any(score < self.relevance_threshold for score in scores)

        # Calculate confidence score (inverse of average distance, normalized)
        avg_score = sum(scores) / len(scores) if scores else 1.0
        # Convert distance to confidence (0 distance = 1.0 confidence)
        confidence = max(0.0, min(1.0, 1.0 - avg_score))

        # Build context from relevant documents
        context_parts = []
        sources = []

        for doc in retrieved_docs:
            score = doc.metadata.get("relevance_score", 1.0)

            # Include document in context
            source_info = f"[Source: {doc.metadata.get('source', 'Unknown')}, Page {doc.metadata.get('page', 'N/A')}]"
            context_parts.append(f"{source_info}\n{doc.page_content}")

            # Create source citation
            sources.append(
                Source(
                    filename=doc.metadata.get("source", "Unknown"),
                    page=doc.metadata.get("page", 0),
                    chunk_index=doc.metadata.get("chunk_index", 0),
                    relevance_score=1.0 - score,  # Convert distance to similarity
                    snippet=doc.page_content[:200] + "..."
                    if len(doc.page_content) > 200
                    else doc.page_content,
                )
            )

        context = "\n\n---\n\n".join(context_parts)

        return {
            "found_in_pdfs": has_relevant,
            "confidence_score": confidence,
            "sources": sources,
            "context": context,
        }

    def _should_generate_answer(self, state: PDFAgentState) -> str:
        """Determine if we should generate an answer or signal not found.

        Args:
            state: Current agent state

        Returns:
            Next node to execute
        """
        if state.get("found_in_pdfs", False):
            return "generate"
        return "not_found"

    def _generate_answer(self, state: PDFAgentState) -> dict:
        """Generate an answer based on retrieved context.

        Args:
            state: Current agent state

        Returns:
            Updated state with generated answer
        """
        query = state["query"]
        context = state.get("context", "")

        chain = PDF_ANSWER_PROMPT | self.llm

        response = chain.invoke(
            {
                "context": context,
                "question": query,
            }
        )

        return {"answer": response.content}

    def _handle_not_found(self, state: PDFAgentState) -> dict:
        """Handle case where no relevant content was found.

        Args:
            state: Current agent state

        Returns:
            Updated state with not-found message
        """
        return {
            "answer": "I could not find this information in the provided documents. "
            "The query may require information not present in the indexed PDFs, "
            "or you may want to try rephrasing your question.",
            "found_in_pdfs": False,
        }

    async def arun(
        self,
        query: str,
        chat_history: list[BaseMessage] | None = None,
    ) -> PDFAgentResult:
        """Run the PDF agent asynchronously.

        Args:
            query: User query to answer
            chat_history: Previous conversation messages

        Returns:
            PDFAgentResult with answer and sources
        """
        initial_state: PDFAgentState = {
            "query": query,
            "chat_history": chat_history or [],
            "retrieved_docs": [],
            "context": "",
            "answer": "",
            "sources": [],
            "confidence_score": 0.0,
            "found_in_pdfs": False,
        }

        result = await self.graph.ainvoke(initial_state)

        return PDFAgentResult(
            answer=result.get("answer", ""),
            sources=result.get("sources", []),
            confidence_score=result.get("confidence_score", 0.0),
            found_in_pdfs=result.get("found_in_pdfs", False),
        )

    def run(
        self,
        query: str,
        chat_history: list[BaseMessage] | None = None,
    ) -> PDFAgentResult:
        """Run the PDF agent synchronously.

        Args:
            query: User query to answer
            chat_history: Previous conversation messages

        Returns:
            PDFAgentResult with answer and sources
        """
        initial_state: PDFAgentState = {
            "query": query,
            "chat_history": chat_history or [],
            "retrieved_docs": [],
            "context": "",
            "answer": "",
            "sources": [],
            "confidence_score": 0.0,
            "found_in_pdfs": False,
        }

        result = self.graph.invoke(initial_state)

        return PDFAgentResult(
            answer=result.get("answer", ""),
            sources=result.get("sources", []),
            confidence_score=result.get("confidence_score", 0.0),
            found_in_pdfs=result.get("found_in_pdfs", False),
        )


def create_pdf_agent(
    model_name: str | None = None,
    relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
    k: int = 5,
) -> PDFAgent:
    """Create a PDF agent instance.

    Args:
        model_name: Optional LLM model name override
        relevance_threshold: Minimum relevance score threshold
        k: Number of documents to retrieve

    Returns:
        Configured PDFAgent
    """
    return PDFAgent(
        model_name=model_name,
        relevance_threshold=relevance_threshold,
        k=k,
    )


# Convenience function for use as a LangGraph node
def pdf_search(
    query: str,
    chat_history: list[BaseMessage] | None = None,
    model_name: str | None = None,
    relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
    k: int = 5,
) -> PDFAgentResult:
    """Search PDFs and answer a query.

    This is a convenience function that can be used directly without
    instantiating the full agent.

    Args:
        query: User query to answer
        chat_history: Previous conversation messages
        model_name: Optional LLM model name override
        relevance_threshold: Minimum relevance score threshold
        k: Number of documents to retrieve

    Returns:
        PDFAgentResult with answer and sources
    """
    agent = create_pdf_agent(
        model_name=model_name,
        relevance_threshold=relevance_threshold,
        k=k,
    )
    return agent.run(query=query, chat_history=chat_history)


# LangGraph node function for integration into larger graphs
def pdf_agent_node(state: dict) -> dict:
    """LangGraph node function for PDF-based question answering.

    This function can be used directly as a node in a LangGraph workflow.
    Expects state to have 'query' and optionally 'chat_history' keys.

    Args:
        state: Graph state containing query and chat_history

    Returns:
        Updated state with answer, sources, and confidence_score
    """
    query = state.get("query", "")
    chat_history = state.get("chat_history", [])

    result = pdf_search(query=query, chat_history=chat_history)

    return {
        "answer": result.answer,
        "sources": [source.model_dump() for source in result.sources],
        "confidence_score": result.confidence_score,
        "found_in_pdfs": result.found_in_pdfs,
    }
