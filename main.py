"""
IDP System v2 - FastAPI Backend
Three-Tier Local Model Architecture for Document Processing

Tier 1: Vision (qwen2.5-vl:7b) - Document understanding
Tier 2: Embedding (nomic-embed-text) - Vector retrieval  
Tier 3: Reasoning (phi-4:14b) - QA synthesis
"""

import logging
import json
import uuid
from datetime import datetime
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from config import settings
from models.schemas import (
    QueryRequest,
    DocumentResponse,
    QueryResponse,
    ProcessingMetrics,
    HealthResponse,
    DocumentStatus,
    ContentType,
    SourceChunk,
    SCHEMA_TEMPLATES
)
from services import (
    VisionService,
    EmbeddingService,
    ReasoningService,
    VectorStoreService,
    ChunkingService
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Service instances (initialized in lifespan)
vision_service: Optional[VisionService] = None
embedding_service: Optional[EmbeddingService] = None
reasoning_service: Optional[ReasoningService] = None
vector_store: Optional[VectorStoreService] = None
chunking_service: Optional[ChunkingService] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global vision_service, embedding_service, reasoning_service
    global vector_store, chunking_service
    
    logger.info("=" * 50)
    logger.info("Initializing IDP System v2")
    logger.info("=" * 50)
    
    vision_service = VisionService()
    embedding_service = EmbeddingService()
    reasoning_service = ReasoningService()
    vector_store = VectorStoreService()
    chunking_service = ChunkingService()
    
    logger.info(f"Vision Model: {settings.VISION_MODEL}")
    logger.info(f"Embedding Model: {settings.EMBEDDING_MODEL}")
    logger.info(f"Reasoning Model: {settings.REASONING_MODEL}")
    logger.info("All services initialized successfully")
    
    yield
    
    logger.info("Shutting down services...")
    if vision_service:
        await vision_service.close()
    if embedding_service:
        await embedding_service.close()
    if reasoning_service:
        await reasoning_service.close()
    logger.info("Shutdown complete")


app = FastAPI(
    title="IDP System v2",
    description="""
## Intelligent Document Processing with Three-Tier Local Model Architecture

### Architecture
- **Tier 1 (Vision)**: qwen2.5-vl:7b - Document understanding & extraction
- **Tier 2 (Embedding)**: nomic-embed-text - Vector embeddings for retrieval
- **Tier 3 (Reasoning)**: phi-4:14b - RAG-based Q&A and synthesis

### Features
- 📄 PDF and image document processing
- 🔍 Natural language document querying
- 📊 Structured data extraction with custom schemas
- 🗄️ Vector-based document storage and retrieval
- 🚀 Fully local - no cloud API dependencies
    """,
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===========================================
# Info & Health Endpoints
# ===========================================

@app.get("/", tags=["Info"])
async def root():
    """API information and available endpoints"""
    return {
        "service": "IDP System v2",
        "version": "2.0.0",
        "architecture": {
            "tier1_vision": settings.VISION_MODEL,
            "tier2_embedding": settings.EMBEDDING_MODEL,
            "tier3_reasoning": settings.REASONING_MODEL
        },
        "endpoints": {
            "process": "POST /api/v1/process",
            "query": "POST /api/v1/query",
            "collections": "GET /api/v1/collections",
            "documents": "GET /api/v1/documents/{collection}",
            "health": "GET /health"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Comprehensive health check for all services"""
    services = {}
    models = {}
    
    # Check Qdrant
    try:
        services["qdrant"] = "operational" if vector_store.health_check() else "error"
    except Exception as e:
        services["qdrant"] = f"error: {str(e)[:50]}"
    
    # Check models via Ollama
    try:
        models["vision"] = await vision_service.check_model_available()
        services["vision"] = "operational" if models["vision"] else "model_missing"
    except:
        models["vision"] = False
        services["vision"] = "error"
    
    try:
        models["embedding"] = await embedding_service.check_model_available()
        services["embedding"] = "operational" if models["embedding"] else "model_missing"
    except:
        models["embedding"] = False
        services["embedding"] = "error"
    
    try:
        models["reasoning"] = await reasoning_service.check_model_available()
        services["reasoning"] = "operational" if models["reasoning"] else "model_missing"
    except:
        models["reasoning"] = False
        services["reasoning"] = "error"
    
    all_healthy = all(v == "operational" for v in services.values())
    
    return HealthResponse(
        status="healthy" if all_healthy else "degraded",
        timestamp=datetime.utcnow(),
        services=services,
        models=models,
        gpu_available=True,  # TODO: actual GPU check
        vram_usage=None
    )


# ===========================================
# Document Processing Endpoints
# ===========================================

@app.post("/api/v1/process", response_model=DocumentResponse, tags=["Processing"])
async def process_document(
    file: UploadFile = File(..., description="PDF or image file to process"),
    collection_name: str = Form(default="documents", description="Target collection"),
    extraction_schema: Optional[str] = Form(default=None, description="JSON schema for extraction"),
    schema_template: Optional[str] = Form(default=None, description="Use template: invoice, resume, contract"),
    index_document: bool = Form(default=True, description="Index for RAG queries")
):
    """
    Process a document through the complete IDP pipeline:
    
    1. **Vision Extraction**: qwen2.5-vl analyzes document layout and extracts content
    2. **Chunking**: Document split into semantic chunks
    3. **Embedding**: nomic-embed-text generates vectors
    4. **Indexing**: Chunks stored in Qdrant for retrieval
    
    Supports: PDF, JPEG, PNG, TIFF, BMP
    """
    start_time = datetime.utcnow()
    document_id = str(uuid.uuid4())
    
    logger.info(f"Processing document: {file.filename} (ID: {document_id})")
    
    # Validate file type
    content_type = file.content_type or ""
    valid_types = [
        "application/pdf",
        "image/jpeg", "image/png", "image/tiff", "image/bmp", "image/jpg"
    ]
    
    if content_type not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {content_type}. Supported: PDF, JPEG, PNG, TIFF, BMP"
        )
    
    try:
        file_content = await file.read()
        
        # Resolve schema
        schema = None
        if schema_template and schema_template in SCHEMA_TEMPLATES:
            schema = SCHEMA_TEMPLATES[schema_template]
        elif extraction_schema:
            try:
                schema = json.loads(extraction_schema)
            except json.JSONDecodeError:
                logger.warning("Invalid JSON schema provided")
        
        # ===== TIER 1: Vision Extraction =====
        vision_start = datetime.utcnow()
        
        if content_type == "application/pdf":
            extracted = await vision_service.extract_from_pdf(
                file_content, schema=schema
            )
        else:
            extracted = await vision_service.extract_from_image(
                file_content, schema=schema
            )
        
        vision_time = (datetime.utcnow() - vision_start).total_seconds()
        logger.info(f"Vision extraction completed in {vision_time:.2f}s")
        
        num_pages = extracted.get("_num_pages", 1)
        chunks_stored = 0
        embedding_time = 0.0
        indexing_time = 0.0
        
        if index_document:
            # ===== CHUNKING =====
            chunks = chunking_service.create_chunks(
                extracted,
                document_id=document_id,
                filename=file.filename
            )
            
            if chunks:
                # ===== TIER 2: Embedding =====
                embed_start = datetime.utcnow()
                
                chunk_texts = [c["text"] for c in chunks]
                embeddings = await embedding_service.generate_document_embeddings(
                    chunk_texts, show_progress=True
                )
                
                embedding_time = (datetime.utcnow() - embed_start).total_seconds()
                logger.info(f"Generated {len(embeddings)} embeddings in {embedding_time:.2f}s")
                
                # ===== INDEXING =====
                index_start = datetime.utcnow()
                
                doc_metadata = {
                    "document_id": document_id,
                    "filename": file.filename,
                    "content_type": content_type,
                    "num_pages": num_pages,
                    "document_type": extracted.get("document_type", "unknown"),
                    "processed_at": datetime.utcnow().isoformat()
                }
                
                vector_store.add_documents(
                    collection_name=collection_name,
                    chunks=chunks,
                    embeddings=embeddings,
                    document_metadata=doc_metadata
                )
                
                indexing_time = (datetime.utcnow() - index_start).total_seconds()
                chunks_stored = len(chunks)
                logger.info(f"Indexed {chunks_stored} chunks in {indexing_time:.2f}s")
        
        total_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Prepare extracted data for response
        extracted_data = {
            "document_type": extracted.get("document_type"),
            "key_value_pairs": extracted.get("key_value_pairs", {}),
            "tables": extracted.get("tables", []),
            "structured_data": extracted.get("structured_data")
        }
        
        return DocumentResponse(
            document_id=document_id,
            status=DocumentStatus.COMPLETED,
            filename=file.filename,
            num_pages=num_pages,
            extracted_data=extracted_data,
            chunks_stored=chunks_stored,
            metrics=ProcessingMetrics(
                total_time=total_time,
                vision_time=vision_time,
                embedding_time=embedding_time,
                indexing_time=indexing_time,
                pages_processed=num_pages,
                chunks_created=chunks_stored
            )
        )
        
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.post("/api/v1/query", response_model=QueryResponse, tags=["Query"])
async def query_documents(request: QueryRequest):
    """
    Query documents using RAG (Retrieval-Augmented Generation)
    
    1. **Embed Query**: nomic-embed-text converts query to vector
    2. **Retrieve**: Qdrant finds most similar chunks
    3. **Reason**: phi-4 synthesizes answer from context
    """
    start_time = datetime.utcnow()
    
    logger.info(f"Query: {request.query[:100]}...")
    
    try:
        # ===== TIER 2: Query Embedding =====
        query_embedding = await embedding_service.generate_query_embedding(request.query)
        
        # ===== RETRIEVAL =====
        filter_conditions = request.filter_metadata or {}
        
        retrieved = vector_store.search(
            collection_name=request.collection_name,
            query_embedding=query_embedding,
            top_k=request.top_k,
            filter_conditions=filter_conditions
        )
        
        if not retrieved:
            return QueryResponse(
                query=request.query,
                answer="No relevant documents found in the collection.",
                sources=[],
                confidence=0.0,
                model_used=settings.REASONING_MODEL,
                processing_time=(datetime.utcnow() - start_time).total_seconds()
            )
        
        # ===== TIER 3: Reasoning =====
        result = await reasoning_service.answer_question(
            query=request.query,
            context_chunks=retrieved,
            include_sources=request.include_sources
        )
        
        # Format sources
        sources = []
        if request.include_sources and "sources" in result:
            for src in result["sources"]:
                sources.append(SourceChunk(
                    text=src.get("text", ""),
                    page_number=src.get("page_number", 0),
                    document_id=src.get("document_id", ""),
                    filename=src.get("filename", "unknown"),
                    score=src.get("score", 0.0),
                    content_type=ContentType(src.get("content_type", "text"))
                ))
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        return QueryResponse(
            query=request.query,
            answer=result["answer"],
            sources=sources,
            confidence=result.get("confidence", 0.5),
            model_used=result.get("model_used", settings.REASONING_MODEL),
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Query failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


# ===========================================
# Collection Management Endpoints
# ===========================================

@app.get("/api/v1/collections", tags=["Collections"])
async def list_collections():
    """List all document collections"""
    try:
        collections = vector_store.list_collections()
        return {"collections": collections}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/collections/{collection_name}", tags=["Collections"])
async def get_collection_info(collection_name: str):
    """Get detailed information about a collection"""
    try:
        info = vector_store.get_collection_info(collection_name)
        documents = vector_store.get_unique_documents(collection_name)
        return {
            "collection": info,
            "documents": documents
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/v1/collections/{collection_name}", tags=["Collections"])
async def delete_collection(collection_name: str):
    """Delete a collection and all its documents"""
    try:
        vector_store.delete_collection(collection_name)
        return {"status": "deleted", "collection": collection_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ===========================================
# Document Management Endpoints
# ===========================================

@app.get("/api/v1/documents/{collection_name}", tags=["Documents"])
async def list_documents(collection_name: str):
    """List all documents in a collection"""
    try:
        documents = vector_store.get_unique_documents(collection_name)
        return {"collection": collection_name, "documents": documents}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/documents/{collection_name}/{document_id}", tags=["Documents"])
async def get_document(collection_name: str, document_id: str):
    """Get document details and chunks"""
    try:
        chunks = vector_store.get_document_chunks(collection_name, document_id)
        if not chunks:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return {
            "document_id": document_id,
            "collection": collection_name,
            "chunk_count": len(chunks),
            "chunks": chunks
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/v1/documents/{collection_name}/{document_id}", tags=["Documents"])
async def delete_document(collection_name: str, document_id: str):
    """Delete a document and all its chunks"""
    try:
        deleted = vector_store.delete_document(collection_name, document_id)
        return {
            "status": "deleted",
            "document_id": document_id,
            "chunks_deleted": deleted
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ===========================================
# Utility Endpoints
# ===========================================

@app.get("/api/v1/schemas", tags=["Utilities"])
async def list_schema_templates():
    """List available extraction schema templates"""
    return {
        "templates": list(SCHEMA_TEMPLATES.keys()),
        "schemas": SCHEMA_TEMPLATES
    }


@app.get("/api/v1/debug/ollama", tags=["Debug"])
async def debug_ollama():
    """Debug endpoint to check Ollama connection and available models"""
    import httpx
    
    result = {
        "ollama_url": settings.OLLAMA_BASE_URL,
        "configured_models": {
            "vision": settings.VISION_MODEL,
            "embedding": settings.EMBEDDING_MODEL,
            "reasoning": settings.REASONING_MODEL
        },
        "ollama_reachable": False,
        "available_models": [],
        "raw_response": None,
        "error": None
    }
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{settings.OLLAMA_BASE_URL}/api/tags")
            result["ollama_reachable"] = response.status_code == 200
            
            if response.status_code == 200:
                data = response.json()
                result["raw_response"] = data
                models = data.get("models", [])
                result["available_models"] = [
                    {"name": m.get("name"), "size": m.get("size")} 
                    for m in models
                ]
            else:
                result["error"] = f"HTTP {response.status_code}: {response.text[:200]}"
                
    except httpx.ConnectError as e:
        result["error"] = f"Connection failed: {str(e)}"
    except Exception as e:
        result["error"] = f"Error: {str(e)}"
    
    return result


@app.post("/api/v1/classify", tags=["Utilities"])
async def classify_document(
    file: UploadFile = File(..., description="Document to classify")
):
    """Quick document classification without full processing"""
    try:
        content = await file.read()
        
        # Use first page only for classification
        if file.content_type == "application/pdf":
            import fitz
            doc = fitz.open(stream=content, filetype="pdf")
            page = doc[0]
            pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
            img_bytes = pix.tobytes("png")
            doc.close()
        else:
            img_bytes = content
        
        result = await vision_service.classify_document(img_bytes)
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ===========================================
# Run Server
# ===========================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )
