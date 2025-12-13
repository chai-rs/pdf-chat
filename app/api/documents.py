"""Document management API routes."""

from datetime import datetime

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.core.dependencies import PDFServiceDep, VectorStoreDep
from app.models.schemas import DocumentInfo, DocumentUploadResponse

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post("/upload", response_model=DocumentUploadResponse)
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
    # Validate file type
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported",
        )

    try:
        # Read file content
        content = await file.read()

        # Process PDF
        documents = pdf_service.process_pdf_bytes(
            content=content,
            filename=file.filename,
        )

        if not documents:
            raise HTTPException(
                status_code=400,
                detail="Could not extract any content from the PDF",
            )

        # Add to vector store
        vector_store.add_documents(documents)

        # Get file hash from first document
        file_hash = documents[0].metadata.get("file_hash", "")

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
        raise HTTPException(
            status_code=500,
            detail=f"Error processing document: {str(e)}",
        )


@router.post("/search")
async def search_documents(
    query: str,
    k: int = 5,
    vector_store: VectorStoreDep = None,
) -> dict:
    """Search for relevant documents.

    Args:
        query: Search query
        k: Number of results
        vector_store: Injected vector store

    Returns:
        Search results
    """
    try:
        results = vector_store.similarity_search_with_score(query, k=k)

        return {
            "query": query,
            "results": [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": float(score),
                }
                for doc, score in results
            ],
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error searching documents: {str(e)}",
        )


@router.delete("/collection")
async def delete_collection(vector_store: VectorStoreDep = None) -> dict:
    """Delete all documents from the collection.

    Args:
        vector_store: Injected vector store

    Returns:
        Success message
    """
    try:
        vector_store.delete_collection()
        return {"message": "Collection deleted successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting collection: {str(e)}",
        )
