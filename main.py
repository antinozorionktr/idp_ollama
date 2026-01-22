"""
Intelligent Document Processing (IDP) System v2.0
FastAPI backend with:
- Hybrid Search (Vector + BM25)
- Retrieve-Rerank Pipeline
- High-DPI OCR (200+)
- Local LLMs (32B-70B via Ollama)
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import uvicorn
import logging
from datetime import datetime
import uuid

from services.ocr_service import OCRService
from services.layout_service import LayoutService
from services.chunking_service import ChunkingService
from services.embedding_service import EmbeddingService
from services.vector_store import VectorStoreService
from services.llm_service import LLMService
from services.reranker_service import RerankerService
from utils.validators import validate_json_output
from config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="IDP System API v2.0",
    description="Intelligent Document Processing with Hybrid Search, Reranking & Local LLMs",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
logger.info("Initializing services...")
ocr_service = OCRService()
layout_service = LayoutService()
chunking_service = ChunkingService()
embedding_service = EmbeddingService()
vector_store = VectorStoreService()
llm_service = LLMService()
reranker_service = RerankerService()
logger.info("All services initialized")


# Request/Response Models
class ProcessDocumentRequest(BaseModel):
    extraction_schema: Optional[Dict[str, Any]] = Field(
        None, 
        description="JSON schema for structured extraction"
    )
    collection_name: Optional[str] = Field(
        "documents", 
        description="Qdrant collection name"
    )
    top_k: Optional[int] = Field(5, description="Number of chunks to retrieve after reranking")
    use_local_llm: Optional[bool] = Field(True, description="Use local LLM (Ollama) vs Cloud API")


class DocumentResponse(BaseModel):
    document_id: str
    status: str
    extracted_data: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any]
    processing_time: float
    chunks_stored: int
    pipeline_version: str = "2.0"


class QueryRequest(BaseModel):
    query: str
    collection_name: Optional[str] = "documents"
    top_k: Optional[int] = 5
    filter_metadata: Optional[Dict[str, Any]] = None
    use_reranking: Optional[bool] = True
    use_hybrid_search: Optional[bool] = True


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    search_type: str  # "hybrid" or "dense"
    reranked: bool


# Health check endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "IDP System",
        "status": "running",
        "version": "2.0.0",
        "features": {
            "hybrid_search": settings.ENABLE_HYBRID_SEARCH,
            "reranking": settings.ENABLE_RERANKING,
            "ocr_dpi": settings.OCR_DPI,
            "llm_provider": settings.LLM_PROVIDER.value,
            "llm_model": settings.OLLAMA_MODEL if settings.LLM_PROVIDER.value == "ollama" else settings.CLAUDE_MODEL
        },
        "endpoints": {
            "process": "/api/v1/process",
            "query": "/api/v1/query",
            "documents": "/api/v1/documents/{collection_name}",
            "schemas": "/api/v1/schemas",
            "collections": "/api/v1/collections",
            "health": "/health",
            "models": "/api/v1/models",
            "config": "/api/v1/config"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check vector store connection
        vector_store.health_check()
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "version": "2.0.0",
            "services": {
                "ocr": f"operational ({settings.OCR_DPI} DPI)",
                "layout_parser": "operational",
                "embeddings": "operational",
                "vector_store": f"operational (hybrid: {settings.ENABLE_HYBRID_SEARCH})",
                "reranker": f"operational" if settings.ENABLE_RERANKING else "disabled",
                "llm": f"operational ({settings.LLM_PROVIDER.value})"
            },
            "config": {
                "hybrid_search": settings.ENABLE_HYBRID_SEARCH,
                "reranking": settings.ENABLE_RERANKING,
                "reranker_model": settings.RERANKER_MODEL if settings.ENABLE_RERANKING else None,
                "llm_model": settings.OLLAMA_MODEL
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Service unhealthy")


@app.post("/api/v1/process", response_model=DocumentResponse)
async def process_document(
    file: UploadFile = File(...),
    extraction_schema: Optional[str] = None,
    collection_name: str = "documents",
    top_k: int = 5,
    use_local_llm: bool = True
):
    """
    Process a document through the complete IDP v2 pipeline:
    1. High-DPI OCR extraction (200+ DPI)
    2. Layout analysis with table detection
    3. Table-aware chunking
    4. Embedding generation
    5. Hybrid vector storage (dense + sparse)
    6. Retrieve candidates
    7. Rerank with cross-encoder
    8. LLM-based extraction (local 32B-70B model)
    9. JSON schema validation
    """
    start_time = datetime.utcnow()
    document_id = str(uuid.uuid4())
    
    logger.info(f"Processing document: {file.filename} (ID: {document_id})")
    
    try:
        # Validate file type
        if not file.content_type in ['application/pdf', 'image/jpeg', 'image/png', 'image/jpg', 'image/tiff']:
            raise HTTPException(
                status_code=400, 
                detail="Invalid file type. Only PDF and images are supported."
            )
        
        # Read file content
        file_content = await file.read()
        
        # Step 1: High-DPI OCR extraction
        logger.info(f"Step 1: High-DPI OCR extraction ({settings.OCR_DPI} DPI) for {document_id}")
        ocr_result = ocr_service.extract_text(
            file_content, 
            file.content_type
        )
        
        # Step 2: Layout analysis
        logger.info(f"Step 2: Layout analysis for {document_id}")
        layout_result = layout_service.analyze_layout(
            file_content,
            file.content_type,
            ocr_result
        )
        
        # Step 3: Table-aware chunking
        logger.info(f"Step 3: Chunking for {document_id}")
        chunks = chunking_service.create_chunks(
            layout_result,
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )
        
        # Step 4: Generate embeddings
        logger.info(f"Step 4: Generating embeddings for {len(chunks)} chunks")
        chunk_texts = [chunk["text"] for chunk in chunks]
        embeddings = embedding_service.generate_embeddings(chunk_texts)
        
        # Step 5: Store in Qdrant with hybrid vectors
        logger.info(f"Step 5: Storing vectors in Qdrant (hybrid: {settings.ENABLE_HYBRID_SEARCH})")
        metadata = {
            "document_id": document_id,
            "filename": file.filename,
            "content_type": file.content_type,
            "processed_at": datetime.utcnow().isoformat(),
            "num_chunks": len(chunks),
            "has_tables": layout_result.get("has_tables", False),
            "ocr_dpi": settings.OCR_DPI,
            "pipeline_version": "2.0"
        }
        
        vector_ids = vector_store.add_documents(
            collection_name=collection_name,
            chunks=chunks,
            embeddings=embeddings,
            metadata=metadata
        )
        
        # Step 6: Retrieve candidates for extraction
        logger.info(f"Step 6: Retrieving candidates (top {settings.RERANK_CANDIDATES})")
        query_text = f"Extract all information from {file.filename}"
        query_embedding = embedding_service.generate_embeddings([query_text])[0]
        
        # Get more candidates for reranking
        candidates = vector_store.search_with_candidates(
            collection_name=collection_name,
            query_embedding=query_embedding,
            query_text=query_text,
            candidates=settings.RERANK_CANDIDATES,
            filter_dict={"document_id": document_id}
        )
        
        # Step 7: Rerank candidates
        logger.info(f"Step 7: Reranking {len(candidates)} candidates to top {top_k}")
        if settings.ENABLE_RERANKING:
            retrieved_chunks = reranker_service.rerank(
                query=query_text,
                documents=candidates,
                top_k=top_k
            )
        else:
            retrieved_chunks = candidates[:top_k]
        
        # Step 8: LLM-based extraction with local model
        logger.info(f"Step 8: LLM extraction using {settings.LLM_PROVIDER.value}")
        context = "\n\n".join([chunk["text"] for chunk in retrieved_chunks])
        
        extracted_data = llm_service.extract_structured_data(
            context=context,
            schema=extraction_schema,
            use_local=use_local_llm
        )
        
        # Step 9: Validate JSON output
        if extraction_schema:
            validated_data = validate_json_output(extracted_data, extraction_schema)
        else:
            validated_data = extracted_data
        
        # Calculate processing time
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        logger.info(f"Document {document_id} processed successfully in {processing_time:.2f}s")
        
        return DocumentResponse(
            document_id=document_id,
            status="completed",
            extracted_data=validated_data,
            metadata=metadata,
            processing_time=processing_time,
            chunks_stored=len(chunks),
            pipeline_version="2.0"
        )
        
    except Exception as e:
        logger.error(f"Error processing document {document_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.post("/api/v1/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query stored documents using Hybrid Search + Reranking RAG
    """
    try:
        logger.info(f"Processing query: {request.query}")
        
        # Generate query embedding
        query_embedding = embedding_service.generate_embeddings([request.query])[0]
        
        # Determine search type
        use_hybrid = request.use_hybrid_search and settings.ENABLE_HYBRID_SEARCH
        
        # Step 1: Retrieve candidates (more than needed for reranking)
        candidates_count = settings.RERANK_CANDIDATES if request.use_reranking else request.top_k
        
        candidates = vector_store.search_with_candidates(
            collection_name=request.collection_name,
            query_embedding=query_embedding,
            query_text=request.query if use_hybrid else None,
            candidates=candidates_count,
            filter_dict=request.filter_metadata
        )
        
        # Step 2: Rerank if enabled
        if request.use_reranking and settings.ENABLE_RERANKING:
            retrieved_chunks = reranker_service.rerank(
                query=request.query,
                documents=candidates,
                top_k=request.top_k
            )
            reranked = True
        else:
            retrieved_chunks = candidates[:request.top_k]
            reranked = False
        
        # Build context from retrieved chunks
        context = "\n\n".join([
            f"[Source: {chunk['metadata'].get('filename', 'unknown')}]\n{chunk['text']}"
            for chunk in retrieved_chunks
        ])
        
        # Generate answer using LLM
        answer = llm_service.generate_answer(
            query=request.query,
            context=context
        )
        
        # Calculate confidence based on relevance scores
        if retrieved_chunks:
            if reranked:
                avg_score = sum(chunk.get("rerank_score", 0) for chunk in retrieved_chunks) / len(retrieved_chunks)
                # Normalize rerank score to 0-1
                avg_score = 1 / (1 + 2.718 ** (-avg_score))
            else:
                avg_score = sum(chunk.get("score", 0) for chunk in retrieved_chunks) / len(retrieved_chunks)
        else:
            avg_score = 0
        
        return QueryResponse(
            query=request.query,
            answer=answer,
            sources=[
                {
                    "text": chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"],
                    "metadata": chunk["metadata"],
                    "score": chunk.get("rerank_score" if reranked else "score", 0),
                    "original_score": chunk.get("original_score", chunk.get("score", 0))
                }
                for chunk in retrieved_chunks
            ],
            confidence=avg_score,
            search_type="hybrid" if use_hybrid else "dense",
            reranked=reranked
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.delete("/api/v1/documents/{document_id}")
async def delete_document(document_id: str, collection_name: str = "documents"):
    """Delete a document and its chunks from the vector store"""
    try:
        deleted_count = vector_store.delete_document(collection_name, document_id)
        return {
            "status": "success",
            "document_id": document_id,
            "deleted_chunks": deleted_count
        }
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Deletion failed: {str(e)}")


@app.get("/api/v1/collections")
async def list_collections():
    """List all available collections in vector store"""
    try:
        collections = vector_store.list_collections()
        return {"collections": collections}
    except Exception as e:
        logger.error(f"Error listing collections: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/documents/{collection_name}")
async def list_documents(collection_name: str, limit: int = 100, offset: int = 0):
    """List all documents in a collection"""
    try:
        # Get collection info
        try:
            collection_info = vector_store.get_collection_info(collection_name)
        except Exception:
            return {
                "collection": collection_name,
                "documents": [],
                "total": 0,
                "message": "Collection not found or empty"
            }
        
        # Scroll through points to get unique document IDs
        documents = {}
        
        # Use scroll to get all points
        scroll_result = vector_store.client.scroll(
            collection_name=collection_name,
            limit=1000,  # Get a batch
            with_payload=True,
            with_vectors=False
        )
        
        points, next_offset = scroll_result
        
        for point in points:
            doc_id = point.payload.get("document_id")
            if doc_id and doc_id not in documents:
                documents[doc_id] = {
                    "document_id": doc_id,
                    "filename": point.payload.get("filename", "unknown"),
                    "file_type": point.payload.get("file_type", "unknown"),
                    "uploaded_at": point.payload.get("uploaded_at", ""),
                    "chunk_count": 0
                }
            if doc_id:
                documents[doc_id]["chunk_count"] += 1
        
        # Convert to list and apply pagination
        doc_list = list(documents.values())
        total = len(doc_list)
        paginated = doc_list[offset:offset + limit]
        
        return {
            "collection": collection_name,
            "documents": paginated,
            "total": total,
            "limit": limit,
            "offset": offset
        }
        
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/schemas")
async def list_schemas():
    """List available extraction schemas"""
    # Default schemas for common document types
    default_schemas = {
        "invoice": {
            "name": "Invoice Schema",
            "description": "Extract data from invoices",
            "schema": {
                "type": "object",
                "properties": {
                    "invoice_number": {"type": "string"},
                    "invoice_date": {"type": "string"},
                    "due_date": {"type": "string"},
                    "vendor_name": {"type": "string"},
                    "vendor_address": {"type": "string"},
                    "customer_name": {"type": "string"},
                    "customer_address": {"type": "string"},
                    "subtotal": {"type": "number"},
                    "tax": {"type": "number"},
                    "total": {"type": "number"},
                    "line_items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "description": {"type": "string"},
                                "quantity": {"type": "number"},
                                "unit_price": {"type": "number"},
                                "amount": {"type": "number"}
                            }
                        }
                    }
                }
            }
        },
        "receipt": {
            "name": "Receipt Schema",
            "description": "Extract data from receipts",
            "schema": {
                "type": "object",
                "properties": {
                    "merchant_name": {"type": "string"},
                    "merchant_address": {"type": "string"},
                    "date": {"type": "string"},
                    "time": {"type": "string"},
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "quantity": {"type": "number"},
                                "price": {"type": "number"}
                            }
                        }
                    },
                    "subtotal": {"type": "number"},
                    "tax": {"type": "number"},
                    "total": {"type": "number"},
                    "payment_method": {"type": "string"}
                }
            }
        },
        "resume": {
            "name": "Resume Schema",
            "description": "Extract data from resumes/CVs",
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "email": {"type": "string"},
                    "phone": {"type": "string"},
                    "location": {"type": "string"},
                    "summary": {"type": "string"},
                    "skills": {"type": "array", "items": {"type": "string"}},
                    "experience": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "company": {"type": "string"},
                                "title": {"type": "string"},
                                "start_date": {"type": "string"},
                                "end_date": {"type": "string"},
                                "description": {"type": "string"}
                            }
                        }
                    },
                    "education": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "institution": {"type": "string"},
                                "degree": {"type": "string"},
                                "field": {"type": "string"},
                                "graduation_date": {"type": "string"}
                            }
                        }
                    }
                }
            }
        },
        "contract": {
            "name": "Contract Schema",
            "description": "Extract key terms from contracts",
            "schema": {
                "type": "object",
                "properties": {
                    "contract_type": {"type": "string"},
                    "parties": {"type": "array", "items": {"type": "string"}},
                    "effective_date": {"type": "string"},
                    "expiration_date": {"type": "string"},
                    "contract_value": {"type": "number"},
                    "payment_terms": {"type": "string"},
                    "key_obligations": {"type": "array", "items": {"type": "string"}},
                    "termination_clause": {"type": "string"},
                    "governing_law": {"type": "string"}
                }
            }
        },
        "general": {
            "name": "General Document",
            "description": "Generic extraction for any document",
            "schema": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "date": {"type": "string"},
                    "author": {"type": "string"},
                    "summary": {"type": "string"},
                    "key_points": {"type": "array", "items": {"type": "string"}},
                    "entities": {
                        "type": "object",
                        "properties": {
                            "people": {"type": "array", "items": {"type": "string"}},
                            "organizations": {"type": "array", "items": {"type": "string"}},
                            "locations": {"type": "array", "items": {"type": "string"}},
                            "dates": {"type": "array", "items": {"type": "string"}},
                            "amounts": {"type": "array", "items": {"type": "string"}}
                        }
                    }
                }
            }
        }
    }
    
    return {
        "schemas": default_schemas,
        "total": len(default_schemas)
    }


@app.get("/api/v1/models")
async def list_models():
    """List available LLM models and current configuration"""
    try:
        return {
            "current_config": llm_service.get_model_info(),
            "available_models": llm_service.list_available_models(),
            "reranker": reranker_service.get_model_info(),
            "embedding_model": settings.EMBEDDING_MODEL
        }
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/config")
async def get_config():
    """Get current system configuration"""
    return {
        "version": "2.0.0",
        "search": {
            "hybrid_enabled": settings.ENABLE_HYBRID_SEARCH,
            "vector_weight": settings.VECTOR_SEARCH_WEIGHT,
            "bm25_weight": 1 - settings.VECTOR_SEARCH_WEIGHT
        },
        "reranking": {
            "enabled": settings.ENABLE_RERANKING,
            "model": settings.RERANKER_MODEL,
            "candidates": settings.RERANK_CANDIDATES,
            "top_k": settings.RERANK_TOP_K
        },
        "ocr": {
            "dpi": settings.OCR_DPI,
            "table_recognition": settings.OCR_ENABLE_TABLE_RECOGNITION
        },
        "llm": {
            "provider": settings.LLM_PROVIDER.value,
            "model": settings.OLLAMA_MODEL,
            "max_tokens": settings.MAX_TOKENS,
            "temperature": settings.TEMPERATURE
        },
        "chunking": {
            "chunk_size": settings.CHUNK_SIZE,
            "chunk_overlap": settings.CHUNK_OVERLAP
        }
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )