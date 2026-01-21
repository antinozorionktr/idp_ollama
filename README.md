# IDP System v2 - Intelligent Document Processing

## Three-Tier Local Model Architecture

A modern document processing system that runs **entirely locally** using state-of-the-art vision and language models through Ollama. No cloud APIs required.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              IDP System v2                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    TIER 1: VISION (The "Eyes")                      │   │
│  │                                                                      │   │
│  │    Model: qwen2.5-vl:7b (~6GB VRAM)                                 │   │
│  │                                                                      │   │
│  │    • Document layout understanding                                   │   │
│  │    • Table detection & extraction                                    │   │
│  │    • Form field recognition                                          │   │
│  │    • Structured JSON output                                          │   │
│  │                                                                      │   │
│  │    Input: PDF/Image → Output: Structured Data                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                         │
│                                    ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                   TIER 2: EMBEDDING (The "Memory")                   │   │
│  │                                                                      │   │
│  │    Model: nomic-embed-text (~500MB)                                 │   │
│  │                                                                      │   │
│  │    • Lightweight & fast                                             │   │
│  │    • Long-context optimized (8192 tokens)                           │   │
│  │    • 768-dimensional vectors                                         │   │
│  │    • Prefix-aware (search_query vs search_document)                 │   │
│  │                                                                      │   │
│  │    Input: Text Chunks → Output: Vectors → Qdrant                    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                         │
│                                    ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                   TIER 3: REASONING (The "Brain")                    │   │
│  │                                                                      │   │
│  │    Model: phi4:14b (~10GB VRAM)                                     │   │
│  │    Alternative: gemma3:12b                                           │   │
│  │                                                                      │   │
│  │    • Superior instruction-following                                  │   │
│  │    • Context-aware reasoning                                         │   │
│  │    • RAG-based answer synthesis                                      │   │
│  │    • Document summarization                                          │   │
│  │                                                                      │   │
│  │    Input: Query + Context → Output: Answer                          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **GPU VRAM** | 12GB | 16GB+ |
| **RAM** | 16GB | 32GB |
| **Storage** | 30GB | 50GB |
| **GPU** | RTX 3060 / 4060 | RTX 3090 / 4080 |

### VRAM Breakdown
- Tier 1 (qwen2.5-vl:7b): ~6GB
- Tier 2 (nomic-embed-text): ~0.5GB
- Tier 3 (phi4:14b): ~10GB

> **Note**: Models are loaded on-demand. Concurrent usage of all tiers requires ~16GB VRAM.

## Quick Start

### Prerequisites

1. **Install Ollama**
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ```

2. **Pull Required Models**
   ```bash
   # Tier 1: Vision
   ollama pull qwen2.5vl:7b
   
   # Tier 2: Embedding
   ollama pull nomic-embed-text
   
   # Tier 3: Reasoning
   ollama pull phi4:14b
   ```

3. **Start Qdrant**
   ```bash
   docker run -d -p 6333:6333 -p 6334:6334 \
     -v qdrant_data:/qdrant/storage \
     qdrant/qdrant
   ```

### Option 1: Local Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Start the server
python main.py
```

### Option 2: Docker Compose

```bash
# Start all services (Ollama, Qdrant, Backend)
docker-compose up -d

# Models will be pulled automatically on first run
# This may take 10-20 minutes depending on connection

# Check status
docker-compose ps
docker-compose logs -f backend
```

### Access Points

- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Qdrant Dashboard**: http://localhost:6333/dashboard

## API Usage

### Process a Document

```bash
# Basic processing
curl -X POST "http://localhost:8000/api/v1/process" \
  -F "file=@invoice.pdf" \
  -F "collection_name=invoices"

# With schema template
curl -X POST "http://localhost:8000/api/v1/process" \
  -F "file=@invoice.pdf" \
  -F "collection_name=invoices" \
  -F "schema_template=invoice"

# With custom schema
curl -X POST "http://localhost:8000/api/v1/process" \
  -F "file=@document.pdf" \
  -F 'extraction_schema={"type":"object","properties":{"title":{"type":"string"}}}'
```

### Query Documents

```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the total amount on the invoice?",
    "collection_name": "invoices",
    "top_k": 5
  }'
```

### Python Client

```python
import httpx

async def process_and_query():
    async with httpx.AsyncClient(base_url="http://localhost:8000") as client:
        # Process document
        with open("invoice.pdf", "rb") as f:
            response = await client.post(
                "/api/v1/process",
                files={"file": f},
                data={"collection_name": "invoices", "schema_template": "invoice"}
            )
        result = response.json()
        print(f"Processed: {result['document_id']}")
        
        # Query
        response = await client.post(
            "/api/v1/query",
            json={
                "query": "What is the invoice total?",
                "collection_name": "invoices"
            }
        )
        answer = response.json()
        print(f"Answer: {answer['answer']}")
```

## Schema Templates

Built-in templates for common document types:

### Invoice Schema
```json
{
  "invoice_number": "string",
  "date": "date",
  "vendor": {"name": "string", "address": "string"},
  "line_items": [{"description": "string", "quantity": "number", "total": "number"}],
  "total": "number"
}
```

### Resume Schema
```json
{
  "name": "string",
  "email": "string",
  "experience": [{"company": "string", "title": "string", "dates": "string"}],
  "education": [{"institution": "string", "degree": "string"}],
  "skills": ["string"]
}
```

### Contract Schema
```json
{
  "title": "string",
  "parties": [{"name": "string", "role": "string"}],
  "effective_date": "date",
  "key_terms": ["string"],
  "total_value": "number"
}
```

## Architecture Comparison

### v1 (Previous) vs v2 (Current)

| Aspect | v1 | v2 |
|--------|----|----|
| **OCR** | PaddleOCR (separate) | qwen2.5-vl (unified) |
| **Layout** | LayoutParser + Detectron2 | qwen2.5-vl (built-in) |
| **Embeddings** | sentence-transformers (local) | nomic-embed-text (Ollama) |
| **LLM** | Claude / GPT-4 (cloud) | phi4 (local) |
| **Dependencies** | Heavy (PyTorch, Detectron2) | Lightweight (Ollama API) |
| **VRAM** | ~8GB | ~16GB (all models loaded) |
| **Privacy** | Data sent to cloud | 100% local |
| **Cost** | API usage fees | One-time hardware |

### Why This Architecture?

1. **Unified Vision Model**: qwen2.5-vl understands layout natively—no need for separate OCR and layout analysis steps.

2. **Lightweight Embeddings**: nomic-embed-text is optimized for retrieval with minimal resources.

3. **Strong Reasoning**: phi4:14b offers GPT-4-class instruction following for RAG tasks.

4. **No Cloud Dependencies**: Complete data privacy—nothing leaves your machine.

5. **Simplified Stack**: Ollama manages all models through a single API.

## Project Structure

```
idp_v2/
├── main.py                 # FastAPI application
├── config.py               # Configuration settings
├── requirements.txt        # Python dependencies
├── Dockerfile              # Backend container
├── docker-compose.yml      # Full stack deployment
│
├── models/
│   ├── __init__.py
│   └── schemas.py          # Pydantic models & templates
│
├── services/
│   ├── __init__.py
│   ├── vision_service.py   # Tier 1: Document understanding
│   ├── embedding_service.py # Tier 2: Vector embeddings
│   ├── reasoning_service.py # Tier 3: QA & synthesis
│   ├── vector_store.py     # Qdrant integration
│   └── chunking_service.py # Document segmentation
│
└── utils/
    └── validators.py       # JSON schema validation
```

## Configuration

Environment variables (`.env` file or system):

```env
# Ollama
OLLAMA_BASE_URL=http://localhost:11434

# Models
VISION_MODEL=qwen2.5-vl:7b
EMBEDDING_MODEL=nomic-embed-text
REASONING_MODEL=phi4:14b

# Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Processing
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
PDF_DPI=150

# Server
HOST=0.0.0.0
PORT=8000
DEBUG=true
```

## Troubleshooting

### Models Not Loading

```bash
# Check Ollama status
curl http://localhost:11434/api/tags

# Pull models manually
ollama pull qwen2.5-vl:7b
ollama pull nomic-embed-text
ollama pull phi4:14b
```

### Out of VRAM

1. Use smaller models:
   - `qwen2.5-vl:3b` instead of 7b
   - `phi3:3.8b` instead of phi4:14b

2. Load models sequentially (modify services to unload after use)

### Slow Processing

- Increase `PDF_DPI` for better quality (but slower)
- Decrease `CHUNK_SIZE` for faster embedding
- Use SSD storage for Qdrant

## License

MIT License - See LICENSE file for details.

---

Built with ❤️ for local, private document intelligence.
