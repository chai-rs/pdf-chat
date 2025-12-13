"""RAG Agent using LangGraph."""

from typing import Annotated, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from app.core.config import settings
from app.db.vector_store import VectorStore
from app.services.web_search import WebSearchService


class AgentState(TypedDict):
    """State for the RAG agent."""

    messages: Annotated[list[BaseMessage], add_messages]
    query: str
    documents: list[Document]
    web_results: list[dict]
    use_web_search: bool
    response: str


class RAGAgent:
    """RAG Agent for Chat with PDF."""

    def __init__(
        self,
        vector_store: VectorStore,
        web_search: WebSearchService | None = None,
        model_name: str | None = None,
    ):
        """Initialize the RAG agent.

        Args:
            vector_store: Vector store for document retrieval
            web_search: Optional web search service
            model_name: LLM model name
        """
        self.vector_store = vector_store
        self.web_search = web_search
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
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("retrieve", self._retrieve_documents)
        workflow.add_node("web_search", self._web_search)
        workflow.add_node("generate", self._generate_response)

        # Add edges
        workflow.add_edge(START, "retrieve")
        workflow.add_conditional_edges(
            "retrieve",
            self._should_web_search,
            {"web_search": "web_search", "generate": "generate"},
        )
        workflow.add_edge("web_search", "generate")
        workflow.add_edge("generate", END)

        return workflow.compile()

    def _retrieve_documents(self, state: AgentState) -> dict:
        """Retrieve relevant documents from vector store."""
        query = state["query"]
        documents = self.vector_store.similarity_search(
            query, k=settings.top_k_results
        )
        return {"documents": documents}

    def _should_web_search(self, state: AgentState) -> str:
        """Determine if web search should be performed."""
        if state.get("use_web_search", False) and self.web_search:
            return "web_search"
        return "generate"

    def _web_search(self, state: AgentState) -> dict:
        """Perform web search for additional context."""
        if not self.web_search:
            return {"web_results": []}

        try:
            results = self.web_search.search(state["query"], max_results=3)
            return {"web_results": results}
        except Exception:
            return {"web_results": []}

    def _generate_response(self, state: AgentState) -> dict:
        """Generate response using retrieved context."""
        # Build context from documents
        doc_context = "\n\n".join(
            f"[Document {i+1}]:\n{doc.page_content}"
            for i, doc in enumerate(state.get("documents", []))
        )

        # Build web context if available
        web_context = ""
        if state.get("web_results"):
            web_parts = []
            for r in state["web_results"]:
                if isinstance(r, dict):
                    title = r.get("title", "N/A")
                    content = r.get("content", r.get("snippet", ""))[:200]
                else:
                    # SearchResult Pydantic model
                    title = r.title or "N/A"
                    content = (r.snippet or "")[:200]
                web_parts.append(f"- {title}: {content}")
            web_context = "\n\n[Web Search Results]:\n" + "\n".join(web_parts)

        # Build prompt
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are a helpful assistant that answers questions based on the provided context.
Use the document context and web search results (if available) to answer the user's question.
If you cannot find the answer in the context, say so clearly.
Always cite your sources when possible.

Context:
{context}
{web_context}""",
            ),
            ("human", "{query}"),
        ])

        # Generate response
        chain = prompt | self.llm
        response = chain.invoke({
            "context": doc_context or "No relevant documents found.",
            "web_context": web_context,
            "query": state["query"],
        })

        return {
            "response": response.content,
            "messages": [AIMessage(content=response.content)],
        }

    async def arun(
        self,
        query: str,
        messages: list[BaseMessage] | None = None,
        use_web_search: bool = False,
    ) -> dict:
        """Run the RAG agent asynchronously.

        Args:
            query: User query
            messages: Previous conversation messages
            use_web_search: Whether to use web search

        Returns:
            Agent output with response and sources
        """
        initial_state: AgentState = {
            "messages": messages or [],
            "query": query,
            "documents": [],
            "web_results": [],
            "use_web_search": use_web_search,
            "response": "",
        }

        # Add human message
        initial_state["messages"].append(HumanMessage(content=query))

        # Run the graph
        result = await self.graph.ainvoke(initial_state)

        return {
            "response": result["response"],
            "documents": result["documents"],
            "web_results": result.get("web_results", []),
        }

    def run(
        self,
        query: str,
        messages: list[BaseMessage] | None = None,
        use_web_search: bool = False,
    ) -> dict:
        """Run the RAG agent synchronously.

        Args:
            query: User query
            messages: Previous conversation messages
            use_web_search: Whether to use web search

        Returns:
            Agent output with response and sources
        """
        initial_state: AgentState = {
            "messages": messages or [],
            "query": query,
            "documents": [],
            "web_results": [],
            "use_web_search": use_web_search,
            "response": "",
        }

        # Add human message
        initial_state["messages"].append(HumanMessage(content=query))

        # Run the graph
        result = self.graph.invoke(initial_state)

        return {
            "response": result["response"],
            "documents": result["documents"],
            "web_results": result.get("web_results", []),
        }


def create_rag_graph(
    vector_store: VectorStore,
    web_search: WebSearchService | None = None,
) -> RAGAgent:
    """Create a RAG agent instance.

    Args:
        vector_store: Vector store for document retrieval
        web_search: Optional web search service

    Returns:
        Configured RAG agent
    """
    return RAGAgent(vector_store=vector_store, web_search=web_search)
