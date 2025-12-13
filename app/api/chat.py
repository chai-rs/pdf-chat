"""Chat API routes."""

import uuid

from fastapi import APIRouter, HTTPException

from app.agents.clarification_agent import create_clarification_agent
from app.agents.rag_agent import create_rag_graph
from app.core.dependencies import VectorStoreDep, WebSearchDep
from app.db.session_memory import session_memory
from app.models.schemas import ChatRequest, ChatResponse

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post("", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    vector_store: VectorStoreDep,
    web_search: WebSearchDep,
) -> ChatResponse:
    """Chat with the PDF documents.

    Args:
        request: Chat request with message and options
        vector_store: Injected vector store
        web_search: Injected web search service

    Returns:
        Chat response with AI answer and sources
    """
    # Get or create session ID
    session_id = request.session_id or str(uuid.uuid4())

    # Get conversation history
    messages = session_memory.get_recent_messages(session_id, n=10)

    try:
        # Check if clarification is needed
        clarification_agent = create_clarification_agent()
        clarification_result = await clarification_agent.arun(
            query=request.message,
            chat_history=messages,
        )

        if clarification_result.needs_clarification:
            # Return clarification request without processing the query
            return ChatResponse(
                response=clarification_result.clarification_question or "Could you please clarify your question?",
                session_id=session_id,
                sources=[],
                web_sources=[],
                needs_clarification=True,
                clarification_question=clarification_result.clarification_question,
            )

        # Create RAG agent
        agent = create_rag_graph(
            vector_store=vector_store,
            web_search=web_search if request.use_web_search else None,
        )

        # Run the agent
        result = await agent.arun(
            query=request.message,
            messages=messages,
            use_web_search=request.use_web_search,
        )

        # Save messages to session
        session_memory.add_human_message(session_id, request.message)
        session_memory.add_ai_message(session_id, result["response"])

        # Format sources
        sources = [
            {
                "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "metadata": doc.metadata,
            }
            for doc in result.get("documents", [])
        ]

        web_sources = []
        for r in result.get("web_results", []):
            if isinstance(r, dict):
                web_sources.append({
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "content": r.get("content", r.get("snippet", ""))[:200],
                })
            else:
                # SearchResult Pydantic model
                web_sources.append({
                    "title": r.title,
                    "url": r.url,
                    "content": r.snippet[:200] if r.snippet else "",
                })

        return ChatResponse(
            response=result["response"],
            session_id=session_id,
            sources=sources,
            web_sources=web_sources,
            needs_clarification=False,
            clarification_question=None,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/session/{session_id}")
async def clear_session(session_id: str) -> dict:
    """Clear a chat session.

    Args:
        session_id: Session ID to clear

    Returns:
        Success message
    """
    session_memory.clear_session(session_id)
    return {"message": f"Session {session_id} cleared"}


@router.get("/session/{session_id}")
async def get_session(session_id: str) -> dict:
    """Get session history.

    Args:
        session_id: Session ID

    Returns:
        Session messages
    """
    messages = session_memory.to_dict(session_id)
    return {
        "session_id": session_id,
        "message_count": len(messages),
        "messages": messages,
    }
