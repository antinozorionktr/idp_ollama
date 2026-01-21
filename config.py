"""
IDP System v2 Configuration
Three-Tier Local Model Architecture
"""

from pydantic_settings import BaseSettings
from typing import Optional
from enum import Enum


class ModelTier(str, Enum):
    """Model tiers for different tasks"""
    VISION = "vision"      # Document parsing & understanding
    EMBEDDING = "embedding" # Vector embeddings for retrieval
    REASONING = "reasoning" # QA and synthesis


class Settings(BaseSettings):
    # ===========================================
    # API Settings
    # ===========================================
    HOST: str = "0.0.0.0"
    PORT: int = 8002
    DEBUG: bool = True
    
    # ===========================================
    # Tier 1: Vision-Language Model (The "Eyes")
    # Model: qwen2.5vl:7b (~6GB VRAM)
    # Purpose: Document layout understanding, table extraction, structured data
    # ===========================================
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    VISION_MODEL: str = "qwen2.5-vl:7b"
    VISION_TEMPERATURE: float = 0.1
    VISION_MAX_TOKENS: int = 4096
    VISION_TIMEOUT: int = 120  # seconds - VLM can be slow on complex docs
    
    # ===========================================
    # Tier 2: Embedding Model (The "Memory")
    # Model: nomic-embed-text
    # Purpose: Lightweight, long-context optimized embeddings
    # ===========================================
    EMBEDDING_MODEL: str = "nomic-embed-text"
    EMBEDDING_DIMENSION: int = 768  # nomic-embed-text dimension
    EMBEDDING_BATCH_SIZE: int = 32
    
    # ===========================================
    # Tier 3: Reasoning Model (The "Brain")
    # Model: phi-4:14b or gemma3:12b (~10-14GB VRAM)
    # Purpose: High-logic QA, synthesis, instruction following
    # ===========================================
    REASONING_MODEL: str = "phi4:14b"  # Alternative: "gemma3:12b"
    REASONING_TEMPERATURE: float = 0.2
    REASONING_MAX_TOKENS: int = 2048
    REASONING_TIMEOUT: int = 60
    
    # ===========================================
    # Vector Store Settings (Qdrant)
    # ===========================================
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_API_KEY: Optional[str] = None
    QDRANT_HTTPS: bool = False
    
    # ===========================================
    # Document Processing Settings
    # ===========================================
    # Chunking
    CHUNK_SIZE: int = 1000  # Larger chunks for better context
    CHUNK_OVERLAP: int = 200
    
    # PDF Processing
    PDF_DPI: int = 150  # Resolution for PDF to image conversion
    MAX_FILE_SIZE: int = 100 * 1024 * 1024  # 100 MB
    ALLOWED_EXTENSIONS: list = [".pdf", ".jpg", ".jpeg", ".png", ".tiff", ".bmp"]
    
    # Processing
    MAX_PAGES_PER_DOC: int = 100
    BATCH_SIZE: int = 4  # Pages to process in parallel
    
    # ===========================================
    # RAG Settings
    # ===========================================
    DEFAULT_TOP_K: int = 5
    SIMILARITY_THRESHOLD: float = 0.5
    
    # ===========================================
    # Optional: Cloud LLM Fallback
    # ===========================================
    ANTHROPIC_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    USE_CLOUD_FALLBACK: bool = False
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
