"""ChromaDB vector store for document embeddings."""

from __future__ import annotations

import os
from typing import Any, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import OpenAIEmbeddings

from app.core.config import settings


# Singleton instance
_chroma_client: Optional[chromadb.PersistentClient] = None
_vector_store: Optional["VectorStore"] = None


def get_chroma_client(persist_directory: str = "./data/chroma") -> chromadb.PersistentClient:
    """Get singleton ChromaDB client.

    Args:
        persist_directory: Directory to persist ChromaDB data

    Returns:
        ChromaDB PersistentClient instance
    """
    global _chroma_client
    if _chroma_client is None:
        os.makedirs(persist_directory, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(
            path=persist_directory,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
    return _chroma_client


def get_vector_store(
    persist_directory: str = "./data/chroma",
    embedding_model: str = "text-embedding-3-small",
    collection_name: str = "pdf_documents",
) -> "VectorStore":
    """Get singleton VectorStore instance.

    Args:
        persist_directory: Directory to persist ChromaDB data
        embedding_model: OpenAI embedding model to use
        collection_name: Name of the ChromaDB collection

    Returns:
        VectorStore singleton instance
    """
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore(
            persist_directory=persist_directory,
            embedding_model=embedding_model,
            collection_name=collection_name,
        )
    return _vector_store


def get_retriever(
    search_type: str = "similarity",
    k: int = 5,
    score_threshold: float = 0.7,
) -> VectorStoreRetriever:
    """Get a LangChain retriever with score threshold filtering.

    Args:
        search_type: Type of search ("similarity" or "similarity_score_threshold")
        k: Number of documents to retrieve
        score_threshold: Minimum relevance score threshold (0.0 to 1.0)

    Returns:
        LangChain VectorStoreRetriever
    """
    store = get_vector_store()
    return store.vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": k,
            "score_threshold": score_threshold,
        },
    )


def search_documents(
    query: str,
    k: int = 5,
    filter: dict[str, Any] | None = None,
) -> list[Document]:
    """Search for documents matching a query.

    Args:
        query: Search query string
        k: Number of documents to return
        filter: Optional metadata filter

    Returns:
        List of Document objects with metadata
    """
    store = get_vector_store()
    results = store.similarity_search_with_score(query, k=k, filter=filter)

    # Add relevance score to document metadata
    documents = []
    for doc, score in results:
        doc.metadata["relevance_score"] = score
        documents.append(doc)

    return documents


def is_result_relevant(score: float, threshold: float = 0.7) -> bool:
    """Determine if a search result is "good enough" based on relevance score.

    Note: ChromaDB returns distance scores where lower is better.
    A threshold of 0.7 means results with distance < 0.7 are considered relevant.

    Args:
        score: The relevance/distance score from the search
        threshold: Maximum distance threshold (lower = more strict)

    Returns:
        True if the result is considered relevant
    """
    # ChromaDB uses distance scores (lower = more similar)
    return score < threshold


class VectorStore:
    """ChromaDB vector store wrapper for RAG operations."""

    def __init__(
        self,
        persist_directory: str = "./data/chroma",
        embedding_model: str = "text-embedding-3-small",
        collection_name: str = "pdf_documents",
    ):
        """Initialize the vector store.

        Args:
            persist_directory: Directory to persist ChromaDB data
            embedding_model: OpenAI embedding model to use
            collection_name: Name of the ChromaDB collection
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        # Use singleton ChromaDB client
        self.client = get_chroma_client(persist_directory)

        # Initialize embeddings with explicit API key and base URL
        self.embeddings = OpenAIEmbeddings(
            model=embedding_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_api_base,
        )

        # Initialize LangChain Chroma wrapper
        self.vectorstore = Chroma(
            client=self.client,
            collection_name=collection_name,
            embedding_function=self.embeddings,
        )

    def add_documents(self, documents: list[Document]) -> list[str]:
        """Add documents to the vector store.

        Args:
            documents: List of LangChain Document objects

        Returns:
            List of document IDs
        """
        return self.vectorstore.add_documents(documents)

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[Document]:
        """Search for similar documents.

        Args:
            query: Search query
            k: Number of results to return
            filter: Optional metadata filter

        Returns:
            List of similar documents
        """
        return self.vectorstore.similarity_search(query, k=k, filter=filter)

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[tuple[Document, float]]:
        """Search for similar documents with relevance scores.

        Args:
            query: Search query
            k: Number of results to return
            filter: Optional metadata filter

        Returns:
            List of (document, score) tuples
        """
        return self.vectorstore.similarity_search_with_score(query, k=k, filter=filter)

    def search_with_relevance_check(
        self,
        query: str,
        k: int = 5,
        threshold: float = 0.7,
        filter: dict[str, Any] | None = None,
    ) -> tuple[list[Document], bool]:
        """Search for documents and determine if results are good enough.

        Args:
            query: Search query
            k: Number of results to return
            threshold: Relevance score threshold
            filter: Optional metadata filter

        Returns:
            Tuple of (documents, has_relevant_results)
        """
        results = self.similarity_search_with_score(query, k=k, filter=filter)

        if not results:
            return [], False

        documents = []
        has_relevant = False

        for doc, score in results:
            doc.metadata["relevance_score"] = score
            documents.append(doc)
            if is_result_relevant(score, threshold):
                has_relevant = True

        return documents, has_relevant

    def delete_collection(self) -> None:
        """Delete the entire collection."""
        self.client.delete_collection(self.collection_name)

    def get_retriever(
        self,
        k: int = 5,
        score_threshold: float | None = None,
    ) -> VectorStoreRetriever:
        """Get a retriever interface for the vector store.

        Args:
            k: Number of documents to retrieve
            score_threshold: Optional minimum relevance score threshold

        Returns:
            LangChain retriever
        """
        if score_threshold is not None:
            return self.vectorstore.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "k": k,
                    "score_threshold": score_threshold,
                },
            )
        return self.vectorstore.as_retriever(search_kwargs={"k": k})
