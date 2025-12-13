# Chat with PDF - Testing Manual

A comprehensive guide for manually testing all components of the Chat with PDF RAG system.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [1. Environment Setup](#1-environment-setup)
- [2. Server Startup](#2-server-startup)
- [3. PDF Ingestion Testing](#3-pdf-ingestion-testing)
- [4. Vector Store Testing](#4-vector-store-testing)
- [5. Conversation Memory Testing](#5-conversation-memory-testing)
- [6. API Endpoint Testing](#6-api-endpoint-testing)
- [7. End-to-End Testing](#7-end-to-end-testing)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager installed
- OpenAI API key (required)
- Tavily API key (optional, for web search)

---

## 1. Environment Setup

### 1.1 Install Dependencies

```bash
cd /Users/0xanonydxck/dev/playground/ingestion
uv sync
```

### 1.2 Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your API keys
# Required: OPENAI_API_KEY
# Optional: TAVILY_API_KEY, ANTHROPIC_API_KEY
```

### 1.3 Verify Configuration

```bash
uv run python -c "
from app.core.config import settings
print('✅ Configuration loaded')
print(f'   LLM Model: {settings.default_llm_model}')
print(f'   Embedding: {settings.embedding_model}')
print(f'   ChromaDB: {settings.chroma_persist_directory}')
"
```

**Expected**: Configuration values printed without errors.

---

## 2. Server Startup

### 2.1 Start Development Server

```bash
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 2.2 Verify Server Health

```bash
# Health check
curl -s http://localhost:8000/health | jq

# Root endpoint
curl -s http://localhost:8000/ | jq
```

**Expected Response**:
```json
{"status": "healthy", "version": "0.1.0"}
```

### 2.3 Access Swagger UI

Open in browser: http://localhost:8000/docs

---

## 3. PDF Ingestion Testing

### 3.1 Using CLI Script

```bash
# Create sample PDF directory
mkdir -p ./data/pdfs

# Place your PDF files in ./data/pdfs, then run:
uv run python scripts/ingest_pdfs.py --pdf-dir ./data/pdfs

# Recursive ingestion
uv run python scripts/ingest_pdfs.py --pdf-dir ./data/pdfs --recursive

# Custom collection
uv run python scripts/ingest_pdfs.py --pdf-dir ./data/pdfs --collection my_docs
```

**Expected Output**:
```
📚 Found X PDF file(s)
📂 Source directory: ./data/pdfs
📁 Collection: pdf_documents
💾 Persist directory: ./data/chroma

📄 Processing: document.pdf
   ✓ Added 15 chunks

✅ Ingestion complete! Total chunks added: 15
```

### 3.2 Using API Upload

```bash
# Upload a single PDF via API
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -F "file=@/path/to/your/document.pdf"
```

**Expected Response**:
```json
{
  "success": true,
  "message": "Successfully processed document.pdf",
  "document": {
    "filename": "document.pdf",
    "file_hash": "abc123...",
    "num_chunks": 15,
    "created_at": "2025-12-11T00:00:00Z"
  }
}
```

### 3.3 Test PDF Service Directly

```bash
uv run python -c "
from pathlib import Path
from app.services.pdf_ingestion import PDFIngestionService

service = PDFIngestionService(chunk_size=1000, chunk_overlap=200)

# Test with a sample PDF (replace with actual path)
# docs = service.process_pdf(Path('./data/pdfs/sample.pdf'))
# print(f'Extracted {len(docs)} chunks')
# print(f'First chunk metadata: {docs[0].metadata}')

print('✅ PDFIngestionService initialized successfully')
"
```

---

## 4. Vector Store Testing

### 4.1 Verify ChromaDB Connection

```bash
uv run python -c "
from app.db.vector_store import get_chroma_client, get_vector_store

# Test singleton client
client = get_chroma_client()
print(f'✅ ChromaDB client connected')
print(f'   Collections: {[c.name for c in client.list_collections()]}')

# Test vector store
store = get_vector_store()
print(f'✅ VectorStore initialized')
print(f'   Collection: {store.collection_name}')
"
```

### 4.2 Test Document Search

```bash
uv run python -c "
from app.db.vector_store import search_documents, get_vector_store

store = get_vector_store()

# Test similarity search (requires ingested documents)
query = 'What is machine learning?'
results = store.similarity_search_with_score(query, k=3)

print(f'Query: {query}')
print(f'Results: {len(results)} documents found')
for doc, score in results:
    print(f'  - Score: {score:.4f}')
    print(f'    Content: {doc.page_content[:100]}...')
    print(f'    Metadata: {doc.metadata}')
"
```

### 4.3 Test Retriever

```bash
uv run python -c "
from app.db.vector_store import get_retriever

retriever = get_retriever(k=5, score_threshold=0.7)
print(f'✅ Retriever configured')
print(f'   Search type: similarity_score_threshold')
print(f'   k: 5, score_threshold: 0.7')
"
```

### 4.4 Search via API

```bash
# Search documents
curl -X POST "http://localhost:8000/api/v1/documents/search?query=machine+learning&k=5" | jq
```

---

## 5. Conversation Memory Testing

### 5.1 Test Memory Module Functions

```bash
uv run python << 'EOF'
import warnings
warnings.filterwarnings("ignore")

from app.db.memory import (
    get_memory, add_message, get_history, clear_memory,
    delete_session, session_exists, get_session_ids
)

print("=" * 50)
print("CONVERSATION MEMORY TEST")
print("=" * 50)

# Test 1: Auto-generate session_id
print("\n1. Auto-generate session_id")
session_id, memory = get_memory(None)
print(f"   Generated: {session_id}")
assert len(session_id) == 36, "UUID should be 36 chars"
print("   ✅ PASS")

# Test 2: Add messages
print("\n2. Add messages")
add_message(session_id, 'human', 'Hello, how are you?')
add_message(session_id, 'ai', 'I am doing well!')
add_message(session_id, 'user', 'Great!')  # alias
add_message(session_id, 'assistant', 'Thanks!')  # alias
history = get_history(session_id)
assert len(history) == 4, f"Expected 4 messages, got {len(history)}"
print(f"   Added 4 messages: {len(history)} in history")
print("   ✅ PASS")

# Test 3: ConversationBufferMemory
print("\n3. ConversationBufferMemory integration")
_, mem = get_memory(session_id)
loaded = mem.load_memory_variables({})
assert "chat_history" in loaded
assert len(loaded["chat_history"]) == 4
print(f"   Memory key 'chat_history': {len(loaded['chat_history'])} messages")
print("   ✅ PASS")

# Test 4: Session management
print("\n4. Session management")
assert session_exists(session_id) == True
assert session_exists("fake-session") == False
sessions = get_session_ids()
assert session_id in sessions
print(f"   session_exists: OK")
print(f"   get_session_ids: {len(sessions)} active")
print("   ✅ PASS")

# Test 5: Clear memory
print("\n5. Clear memory")
result = clear_memory(session_id)
assert result == True
assert len(get_history(session_id)) == 0
assert session_exists(session_id) == True  # session still exists
print("   Cleared messages, session preserved")
print("   ✅ PASS")

# Test 6: Delete session
print("\n6. Delete session")
result = delete_session(session_id)
assert result == True
assert session_exists(session_id) == False
print("   Session deleted completely")
print("   ✅ PASS")

# Test 7: History truncation
print("\n7. History truncation (max 10 pairs = 20 messages)")
test_session = "truncation-test"
for i in range(15):
    add_message(test_session, 'human', f'Question {i+1}')
    add_message(test_session, 'ai', f'Answer {i+1}')
history = get_history(test_session)
assert len(history) == 20, f"Expected 20, got {len(history)}"
assert "Question 6" in history[0].content, f"First should be Q6, got {history[0].content}"
assert "Answer 15" in history[-1].content, f"Last should be A15, got {history[-1].content}"
delete_session(test_session)
print(f"   Added 30 messages, retained 20 (last 10 pairs)")
print("   ✅ PASS")

# Test 8: Thread safety
print("\n8. Thread safety (basic)")
import threading
errors = []
def worker(sid, n):
    try:
        for i in range(n):
            add_message(sid, 'human', f'msg-{i}')
    except Exception as e:
        errors.append(e)

threads = [threading.Thread(target=worker, args=('thread-test', 10)) for _ in range(5)]
for t in threads: t.start()
for t in threads: t.join()
assert len(errors) == 0, f"Thread errors: {errors}"
delete_session('thread-test')
print("   5 threads × 10 messages: No errors")
print("   ✅ PASS")

print("\n" + "=" * 50)
print("ALL TESTS PASSED ✅")
print("=" * 50)
EOF
```

### 5.2 Test via API

```bash
# Get session history
curl -s http://localhost:8000/api/v1/chat/session/test-session-123 | jq

# Clear session
curl -X DELETE http://localhost:8000/api/v1/chat/session/test-session-123 | jq
```

---

## 6. API Endpoint Testing

### 6.1 Chat Endpoint

```bash
# Basic chat (requires ingested PDFs)
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What are the main topics in the documents?",
    "session_id": null
  }' | jq

# Chat with specific session
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Tell me more about that",
    "session_id": "my-session-123"
  }' | jq

# Chat with web search enabled
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is the latest news about AI?",
    "session_id": null,
    "use_web_search": true
  }' | jq
```

**Expected Response**:
```json
{
  "response": "Based on the documents...",
  "session_id": "abc123-...",
  "sources": [
    {
      "content": "...",
      "metadata": {"source": "document.pdf", "page": 1}
    }
  ],
  "web_sources": []
}
```

### 6.2 Documents Endpoints

```bash
# Upload document
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -F "file=@document.pdf"

# Search documents
curl -X POST "http://localhost:8000/api/v1/documents/search?query=python&k=5" | jq

# Delete collection (⚠️ destructive!)
curl -X DELETE http://localhost:8000/api/v1/documents/collection | jq
```

### 6.3 Session Endpoints

```bash
# Get session history
curl -s http://localhost:8000/api/v1/chat/session/{session_id} | jq

# Clear session
curl -X DELETE http://localhost:8000/api/v1/chat/session/{session_id} | jq
```

---

## 7. End-to-End Testing

### 7.1 Complete Workflow

```bash
# Step 1: Check health
curl -s http://localhost:8000/health | jq

# Step 2: Ingest a PDF
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -F "file=@./data/pdfs/sample.pdf" | jq

# Step 3: Search for content
curl -X POST "http://localhost:8000/api/v1/documents/search?query=introduction&k=3" | jq

# Step 4: Start a chat conversation
SESSION_ID=$(curl -s -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Summarize the main points"}' | jq -r '.session_id')

echo "Session ID: $SESSION_ID"

# Step 5: Continue conversation
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d "{\"message\": \"What else is mentioned?\", \"session_id\": \"$SESSION_ID\"}" | jq

# Step 6: Check session history
curl -s "http://localhost:8000/api/v1/chat/session/$SESSION_ID" | jq

# Step 7: Clear session
curl -X DELETE "http://localhost:8000/api/v1/chat/session/$SESSION_ID" | jq
```

### 7.2 Load Testing (Optional)

```bash
# Install hey load testing tool
# brew install hey

# Test health endpoint under load
hey -n 100 -c 10 http://localhost:8000/health

# Test search endpoint
hey -n 50 -c 5 -m POST \
  "http://localhost:8000/api/v1/documents/search?query=test&k=3"
```

---

## Troubleshooting

### Common Issues

| Issue                        | Solution                                                     |
| ---------------------------- | ------------------------------------------------------------ |
| `ModuleNotFoundError`        | Run `uv sync` to install dependencies                        |
| `OPENAI_API_KEY not set`     | Check `.env` file configuration                              |
| `ChromaDB connection error`  | Ensure `./data/chroma` directory exists and is writable      |
| `No documents found`         | Ingest PDFs first using the CLI script                       |
| `Memory deprecation warning` | Safe to ignore - using `langchain_classic` for compatibility |

### Check Logs

```bash
# Run server with debug logging
LOG_LEVEL=debug uv run uvicorn app.main:app --reload

# Check ChromaDB data
ls -la ./data/chroma/
```

### Reset State

```bash
# Clear all ChromaDB data
rm -rf ./data/chroma/

# Clear session data (in-memory, resets on server restart)
# Or use API: DELETE /api/v1/chat/session/{session_id}
```

---

## Quick Reference

| Component     | Test Command                                                              |
| ------------- | ------------------------------------------------------------------------- |
| Server        | `curl http://localhost:8000/health`                                       |
| PDF Ingestion | `uv run python scripts/ingest_pdfs.py --pdf-dir ./data/pdfs`              |
| Vector Store  | `curl -X POST "http://localhost:8000/api/v1/documents/search?query=test"` |
| Chat          | `curl -X POST http://localhost:8000/api/v1/chat -d '{"message":"hello"}'` |
| Memory        | See [Section 5.1](#51-test-memory-module-functions)                       |

---

*Last updated: 2025-12-11*
