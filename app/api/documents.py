"""Document management API routes."""

import logging
from datetime import datetime

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

from app.core.dependencies import PDFServiceDep, VectorStoreDep


logger = logging.getLogger(__name__)

router = APIRouter(tags=["Documents"])


# --- Request/Response Models ---


class DocumentInfo(BaseModel):
    """Document information model."""

    filename: str = Field(description="Original filename")
    file_hash: str = Field(description="MD5 hash of the file")
    num_chunks: int = Field(description="Number of chunks created")
    created_at: datetime = Field(description="Upload timestamp")


class DocumentUploadResponse(BaseModel):
    """Document upload response model."""

    success: bool = Field(description="Upload success status")
    message: str = Field(description="Status message")
    document: DocumentInfo | None = Field(default=None, description="Document info")


class SearchResult(BaseModel):
    """Individual search result."""

    content: str = Field(description="Document content")
    metadata: dict = Field(default_factory=dict, description="Document metadata")
    score: float = Field(description="Relevance score")


class SearchResponse(BaseModel):
    """Search results response."""

    query: str = Field(description="Original query")
    results: list[SearchResult] = Field(default_factory=list, description="Search results")
    count: int = Field(description="Number of results")


# --- Endpoints ---


@router.post("/upload", response_model=DocumentUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    pdf_service: PDFServiceDep = None,
    vector_store: VectorStoreDep = None,
) -> DocumentUploadResponse:
    """Upload and process a PDF document.

    Args:
        file: PDF file to upload
        pdf_service: Injected PDF service
        vector_store: Injected vector store

    Returns:
        Upload response with document info
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported",
        )

    logger.info(f"Uploading document: {file.filename}")

    try:
        content = await file.read()

        documents = pdf_service.process_pdf_bytes(
            content=content,
            filename=file.filename,
        )

        if not documents:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Could not extract any content from the PDF",
            )

        vector_store.add_documents(documents)

        file_hash = documents[0].metadata.get("file_hash", "")

        logger.info(f"Successfully processed {file.filename} into {len(documents)} chunks")

        return DocumentUploadResponse(
            success=True,
            message=f"Successfully processed {file.filename}",
            document=DocumentInfo(
                filename=file.filename,
                file_hash=file_hash,
                num_chunks=len(documents),
                created_at=datetime.utcnow(),
            ),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error processing document: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing document: {str(e)}",
        )


@router.post("/search", response_model=SearchResponse, status_code=status.HTTP_200_OK)
async def search_documents(
    query: str,
    k: int = 5,
    vector_store: VectorStoreDep = None,
) -> SearchResponse:
    """Search for relevant documents.

    Args:
        query: Search query
        k: Number of results to return
        vector_store: Injected vector store

    Returns:
        Search results with relevance scores
    """
    logger.info(f"Searching documents: '{query}' (k={k})")

    try:
        results = vector_store.similarity_search_with_score(query, k=k)

        search_results = [
            SearchResult(
                content=doc.page_content,
                metadata=doc.metadata,
                score=float(score),
            )
            for doc, score in results
        ]

        return SearchResponse(
            query=query,
            results=search_results,
            count=len(search_results),
        )

    except Exception as e:
        logger.exception(f"Error searching documents: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error searching documents: {str(e)}",
        )


@router.delete("/collection", status_code=status.HTTP_200_OK)
async def delete_collection(vector_store: VectorStoreDep = None) -> dict:
    """Delete all documents from the collection.

    Warning: This action is irreversible!
    """
    logger.warning("Deleting entire document collection")

    try:
        vector_store.delete_collection()
        return {"message": "Collection deleted successfully"}
    except Exception as e:
        logger.exception(f"Error deleting collection: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting collection: {str(e)}",
        )
