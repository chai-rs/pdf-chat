"""PDF ingestion service for parsing and chunking PDFs."""

import hashlib
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class PDFIngestionService:
    """Service for ingesting PDF documents."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        """Initialize the PDF ingestion service.

        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def load_pdf(self, file_path: str | Path) -> list[Document]:
        """Load a PDF file and return documents.

        Args:
            file_path: Path to the PDF file

        Returns:
            List of Document objects (one per page)
        """
        loader = PyPDFLoader(str(file_path))
        return loader.load()

    def chunk_documents(self, documents: list[Document]) -> list[Document]:
        """Split documents into smaller chunks.

        Args:
            documents: List of documents to chunk

        Returns:
            List of chunked documents with chunk_index metadata
        """
        chunks = self.text_splitter.split_documents(documents)
        # Add chunk_index to metadata for each chunk
        for idx, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = idx
        return chunks

    def process_pdf(
        self,
        file_path: str | Path,
        metadata: dict | None = None,
    ) -> list[Document]:
        """Load and chunk a PDF file.

        Args:
            file_path: Path to the PDF file
            metadata: Additional metadata to add to documents

        Returns:
            List of chunked documents with metadata
        """
        file_path = Path(file_path)

        # Load PDF
        documents = self.load_pdf(file_path)

        # Add file metadata
        file_hash = self._compute_file_hash(file_path)
        for doc in documents:
            doc.metadata.update(
                {
                    "source": str(file_path),
                    "filename": file_path.name,
                    "file_hash": file_hash,
                }
            )
            if metadata:
                doc.metadata.update(metadata)

        # Chunk documents
        return self.chunk_documents(documents)

    def process_pdf_bytes(
        self,
        content: bytes,
        filename: str,
        metadata: dict | None = None,
    ) -> list[Document]:
        """Process PDF from bytes content.

        Args:
            content: PDF file content as bytes
            filename: Original filename
            metadata: Additional metadata

        Returns:
            List of chunked documents
        """
        import tempfile

        # Write to temporary file
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)

        try:
            # Process the temporary file
            documents = self.process_pdf(tmp_path, metadata)

            # Update filename in metadata
            for doc in documents:
                doc.metadata["filename"] = filename
                doc.metadata["source"] = filename

            return documents
        finally:
            # Clean up temporary file
            tmp_path.unlink()

    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute MD5 hash of a file.

        Args:
            file_path: Path to the file

        Returns:
            MD5 hash string
        """
        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
