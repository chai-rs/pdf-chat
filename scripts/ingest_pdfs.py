#!/usr/bin/env python3
"""Script to ingest PDF files into the vector store."""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

from app.core.config import settings
from app.db.vector_store import VectorStore
from app.services.pdf_ingestion import PDFIngestionService


def ingest_pdfs(pdf_paths: list[Path], collection_name: str = "pdf_documents") -> None:
    """Ingest PDF files into the vector store.

    Args:
        pdf_paths: List of paths to PDF files
        collection_name: Name of the ChromaDB collection
    """
    # Initialize services
    pdf_service = PDFIngestionService(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    vector_store = VectorStore(
        persist_directory=settings.chroma_persist_directory,
        embedding_model=settings.embedding_model,
        collection_name=collection_name,
    )

    total_chunks = 0

    for pdf_path in pdf_paths:
        if not pdf_path.exists():
            print(f"⚠️  File not found: {pdf_path}")
            continue

        if not pdf_path.suffix.lower() == ".pdf":
            print(f"⚠️  Skipping non-PDF file: {pdf_path}")
            continue

        print(f"📄 Processing: {pdf_path.name}")

        try:
            # Process PDF
            documents = pdf_service.process_pdf(pdf_path)

            if not documents:
                print(f"⚠️  No content extracted from: {pdf_path.name}")
                continue

            # Add to vector store
            vector_store.add_documents(documents)

            print(f"   ✓ Added {len(documents)} chunks")
            total_chunks += len(documents)

        except Exception as e:
            print(f"❌ Error processing {pdf_path.name}: {e}")

    print(f"\n✅ Ingestion complete! Total chunks added: {total_chunks}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Ingest PDF files into the vector store")
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=Path("./data/pdfs"),
        help="Directory containing PDF files (default: ./data/pdfs)",
    )
    parser.add_argument(
        "--collection",
        "-c",
        default="pdf_documents",
        help="ChromaDB collection name (default: pdf_documents)",
    )
    parser.add_argument(
        "--recursive",
        "-r",
        action="store_true",
        help="Recursively search directories for PDFs",
    )

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    # Ensure the PDF directory exists
    pdf_dir = Path(args.pdf_dir)
    if not pdf_dir.exists():
        print(f"📁 Creating directory: {pdf_dir}")
        pdf_dir.mkdir(parents=True, exist_ok=True)

    # Collect PDF files from directory
    pdf_files: list[Path] = []
    pattern = "**/*.pdf" if args.recursive else "*.pdf"
    pdf_files.extend(pdf_dir.glob(pattern))

    if not pdf_files:
        print(f"📭 No PDF files found in: {pdf_dir}")
        print("   Add PDF files to this directory and run again.")
        sys.exit(0)

    print(f"📚 Found {len(pdf_files)} PDF file(s)")
    print(f"📂 Source directory: {pdf_dir}")
    print(f"📁 Collection: {args.collection}")
    print(f"💾 Persist directory: {settings.chroma_persist_directory}\n")

    ingest_pdfs(pdf_files, collection_name=args.collection)


if __name__ == "__main__":
    main()
