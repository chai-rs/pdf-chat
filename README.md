# Chat with PDF

A RAG (Retrieval-Augmented Generation) system for chatting with PDF documents using FastAPI, LangGraph, and ChromaDB.

## Features

- 📄 **PDF Ingestion**: Upload and process PDF documents with automatic chunking
- 🔍 **Semantic Search**: ChromaDB vector store for similarity search
- 🤖 **RAG Agent**: LangGraph-powered agent for intelligent responses
- 🌐 **Web Search**: Optional Tavily integration for web-augmented answers
- 💬 **Session Memory**: Conversation history management
- 🐳 **Docker Ready**: Multi-stage Dockerfile with docker-compose

## Tech Stack

- **FastAPI** - Modern Python web framework
- **LangChain & LangGraph** - LLM orchestration
- **ChromaDB** - Vector database
- **OpenAI / Anthropic** - LLM providers
- **Tavily** - Web search
- **uv** - Fast Python package manager

## Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- OpenAI API key (or Anthropic)

### Installation

```bash
# Clone the repository
cd chat-with-pdf

# Install dependencies with uv
uv sync

# Copy environment file and add your API keys
cp .env.example .env
```

### Configuration

Edit `.env` with your API keys:

```env
OPENAI_API_KEY=sk-your-openai-key
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key  # Optional
TAVILY_API_KEY=tvly-your-tavily-key          # Optional for web search
```

### Running the Server

```bash
# Development mode
uv run uvicorn app.main:app --reload

# Production mode
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`. Visit `http://localhost:8000/docs` for the Swagger UI.

### Ingesting PDFs

```bash
# Single file
uv run python scripts/ingest_pdfs.py path/to/document.pdf

# Multiple files
uv run python scripts/ingest_pdfs.py doc1.pdf doc2.pdf doc3.pdf

# Directory (recursive)
uv run python scripts/ingest_pdfs.py -r path/to/pdf/directory
```

## API Endpoints

### Chat

```bash
# Send a message
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the main topic of the documents?"}'

# With web search enabled
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the latest news about AI?", "use_web_search": true}'
```

### Documents

```bash
# Upload a PDF
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -F "file=@document.pdf"

# Search documents
curl -X POST "http://localhost:8000/api/v1/documents/search?query=machine+learning&k=5"
```

### Health Check

```bash
curl http://localhost:8000/health
```

## Docker

### Build and Run

```bash
# Build the image
docker compose build

# Run the container
docker compose up -d

# View logs
docker compose logs -f
```

## Project Structure

```
.
├── app/
│   ├── agents/          # LangGraph agents
│   │   └── rag_agent.py
│   ├── api/             # FastAPI routes
│   │   ├── chat.py
│   │   └── documents.py
│   ├── core/            # Config, dependencies
│   │   ├── config.py
│   │   └── dependencies.py
│   ├── db/              # Vector store, session memory
│   │   ├── vector_store.py
│   │   └── session_memory.py
│   ├── models/          # Pydantic schemas
│   │   └── schemas.py
│   ├── services/        # PDF ingestion, web search
│   │   ├── pdf_ingestion.py
│   │   └── web_search.py
│   └── main.py
├── scripts/
│   └── ingest_pdfs.py
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## Development

```bash
# Install dev dependencies
uv sync --dev

# Run tests
uv run pytest

# Run linter
uv run ruff check .

# Format code
uv run ruff format .
```

## License

MIT
