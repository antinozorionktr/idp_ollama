"""
Pydantic Models for IDP System v2
Request/Response schemas and internal data models
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum


# ===========================================
# Enums
# ===========================================

class DocumentStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ContentType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    FIGURE = "figure"
    HEADER = "header"
    FOOTER = "footer"
    LIST = "list"


# ===========================================
# Internal Data Models
# ===========================================

class BoundingBox(BaseModel):
    """Bounding box for layout elements"""
    x1: float
    y1: float
    x2: float
    y2: float
    
    @property
    def width(self) -> float:
        return self.x2 - self.x1
    
    @property
    def height(self) -> float:
        return self.y2 - self.y1


class PageContent(BaseModel):
    """Extracted content from a single page"""
    page_number: int
    raw_text: str
    structured_data: Optional[Dict[str, Any]] = None
    tables: List[Dict[str, Any]] = Field(default_factory=list)
    figures: List[Dict[str, Any]] = Field(default_factory=list)
    layout_elements: List[Dict[str, Any]] = Field(default_factory=list)
    confidence: float = 1.0


class DocumentChunk(BaseModel):
    """A chunk of document content for indexing"""
    chunk_id: str
    text: str
    content_type: ContentType = ContentType.TEXT
    page_number: int
    document_id: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[List[float]] = None


class ExtractedDocument(BaseModel):
    """Complete extracted document data"""
    document_id: str
    filename: str
    num_pages: int
    pages: List[PageContent]
    full_text: str
    structured_output: Optional[Dict[str, Any]] = None
    chunks: List[DocumentChunk] = Field(default_factory=list)
    processing_time: float
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ===========================================
# API Request Models
# ===========================================

class ProcessDocumentRequest(BaseModel):
    """Request to process a document"""
    extraction_schema: Optional[Dict[str, Any]] = Field(
        None,
        description="JSON schema defining the structure of data to extract"
    )
    collection_name: str = Field(
        default="documents",
        description="Qdrant collection to store the document"
    )
    extraction_prompt: Optional[str] = Field(
        None,
        description="Custom prompt for the vision model"
    )
    index_document: bool = Field(
        default=True,
        description="Whether to index the document for RAG queries"
    )


class QueryRequest(BaseModel):
    """Request to query documents"""
    query: str = Field(..., description="Natural language question")
    collection_name: str = Field(
        default="documents",
        description="Collection to search"
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of relevant chunks to retrieve"
    )
    filter_metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Metadata filters for search"
    )
    include_sources: bool = Field(
        default=True,
        description="Whether to include source chunks in response"
    )


class BatchProcessRequest(BaseModel):
    """Request to process multiple documents"""
    collection_name: str = "documents"
    extraction_schema: Optional[Dict[str, Any]] = None


# ===========================================
# API Response Models
# ===========================================

class ProcessingMetrics(BaseModel):
    """Detailed processing metrics"""
    total_time: float
    vision_time: float
    embedding_time: float
    indexing_time: float
    pages_processed: int
    chunks_created: int


class DocumentResponse(BaseModel):
    """Response after processing a document"""
    document_id: str
    status: DocumentStatus
    filename: str
    num_pages: int
    extracted_data: Optional[Dict[str, Any]] = None
    chunks_stored: int
    metrics: ProcessingMetrics
    message: Optional[str] = None


class SourceChunk(BaseModel):
    """A source chunk returned in query results"""
    text: str
    page_number: int
    document_id: str
    filename: str
    score: float
    content_type: ContentType


class QueryResponse(BaseModel):
    """Response to a document query"""
    query: str
    answer: str
    sources: List[SourceChunk]
    confidence: float
    model_used: str
    processing_time: float


class CollectionInfo(BaseModel):
    """Information about a vector collection"""
    name: str
    document_count: int
    chunk_count: int
    created_at: Optional[datetime] = None


class HealthResponse(BaseModel):
    """System health check response"""
    status: str
    timestamp: datetime
    services: Dict[str, str]
    models: Dict[str, bool]
    gpu_available: bool
    vram_usage: Optional[Dict[str, Any]] = None


# ===========================================
# Schema Templates (Common Document Types)
# ===========================================

INVOICE_SCHEMA = {
    "type": "object",
    "properties": {
        "invoice_number": {"type": "string"},
        "date": {"type": "string", "format": "date"},
        "due_date": {"type": "string", "format": "date"},
        "vendor": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "address": {"type": "string"},
                "tax_id": {"type": "string"}
            }
        },
        "customer": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "address": {"type": "string"}
            }
        },
        "line_items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "quantity": {"type": "number"},
                    "unit_price": {"type": "number"},
                    "total": {"type": "number"}
                }
            }
        },
        "subtotal": {"type": "number"},
        "tax": {"type": "number"},
        "total": {"type": "number"},
        "currency": {"type": "string"}
    }
}

RESUME_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "email": {"type": "string"},
        "phone": {"type": "string"},
        "location": {"type": "string"},
        "summary": {"type": "string"},
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
                    "year": {"type": "string"}
                }
            }
        },
        "skills": {"type": "array", "items": {"type": "string"}}
    }
}

CONTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "parties": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "role": {"type": "string"},
                    "address": {"type": "string"}
                }
            }
        },
        "effective_date": {"type": "string", "format": "date"},
        "termination_date": {"type": "string", "format": "date"},
        "key_terms": {"type": "array", "items": {"type": "string"}},
        "obligations": {"type": "array", "items": {"type": "string"}},
        "payment_terms": {"type": "string"},
        "total_value": {"type": "number"},
        "currency": {"type": "string"}
    }
}

SCHEMA_TEMPLATES = {
    "invoice": INVOICE_SCHEMA,
    "resume": RESUME_SCHEMA,
    "contract": CONTRACT_SCHEMA
}
