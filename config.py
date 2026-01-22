"""
Configuration settings for IDP System v2.0 (Upgraded)
- Hybrid Search (Vector + BM25)
- Reranking
- High-DPI Vision (200+)
- Large Local LLMs (32B-70B)
"""

from pydantic_settings import BaseSettings
from typing import Optional, Literal
from enum import Enum


class LLMProvider(str, Enum):
    OLLAMA = "ollama"
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    VLLM = "vllm"


class Settings(BaseSettings):
    # API Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True
    
    # ==================== VECTOR STORE (Qdrant) ====================
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_API_KEY: Optional[str] = None
    QDRANT_HTTPS: bool = False
    
    # ==================== EMBEDDING SETTINGS ====================
    EMBEDDING_MODEL: str = "BAAI/bge-large-en-v1.5"
    EMBEDDING_DIMENSION: int = 1024
    EMBEDDING_DEVICE: str = "cuda"  # or "cpu"
    
    # ==================== HYBRID SEARCH (NEW) ====================
    # Enable hybrid search (Vector + BM25 keyword search)
    ENABLE_HYBRID_SEARCH: bool = True
    # Weight for vector search (1 - this = BM25 weight)
    VECTOR_SEARCH_WEIGHT: float = 0.7  # 70% vector, 30% BM25
    # BM25 parameters
    BM25_K1: float = 1.5  # Term frequency saturation
    BM25_B: float = 0.75  # Length normalization
    
    # ==================== RERANKING (NEW) ====================
    ENABLE_RERANKING: bool = True
    # Reranker model - cross-encoder for better relevance
    RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"  # or "cross-encoder/ms-marco-MiniLM-L-12-v2"
    RERANKER_DEVICE: str = "cuda"
    # Retrieve more candidates, then rerank to top_k
    RERANK_CANDIDATES: int = 20  # Retrieve 20, rerank to top 5
    RERANK_TOP_K: int = 5
    
    # ==================== CHUNKING SETTINGS ====================
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    
    # ==================== OCR SETTINGS (UPGRADED) ====================
    PADDLE_OCR_LANG: str = "en"
    PADDLE_OCR_USE_GPU: bool = True
    # UPGRADED: Higher DPI for better text extraction
    OCR_DPI: int = 200  # Upgraded from 100 to 200+
    OCR_ENABLE_TABLE_RECOGNITION: bool = True
    OCR_ENABLE_LAYOUT_ANALYSIS: bool = True
    
    # ==================== LAYOUT PARSER SETTINGS ====================
    LAYOUT_MODEL: str = "lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config"
    LAYOUT_CONFIDENCE_THRESHOLD: float = 0.5
    
    # ==================== LLM SETTINGS (UPGRADED) ====================
    # Primary LLM Provider
    LLM_PROVIDER: LLMProvider = LLMProvider.OLLAMA
    
    # Ollama Settings (Local LLMs - 32B to 70B)
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "deepseek-r1:32b"  # Options: deepseek-r1:32b, llama3.3:70b, qwen2.5:32b
    OLLAMA_TIMEOUT: int = 300  # 5 minutes for large models
    
    # Alternative: vLLM for high-throughput inference
    VLLM_BASE_URL: str = "http://localhost:8080"
    VLLM_MODEL: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    
    # Cloud API fallback
    ANTHROPIC_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    CLAUDE_MODEL: str = "claude-sonnet-4-20250514"
    GPT_MODEL: str = "gpt-4-turbo-preview"
    
    # LLM Generation Settings
    MAX_TOKENS: int = 8192  # Increased for complex extractions
    TEMPERATURE: float = 0.1
    
    # ==================== VISION LLM (NEW) ====================
    # For direct image understanding without OCR
    ENABLE_VISION_LLM: bool = True
    VISION_MODEL: str = "llama3.2-vision:11b"  # or "llava:34b"
    VISION_DPI: int = 300  # High DPI for vision models
    
    # ==================== FILE UPLOAD SETTINGS ====================
    MAX_FILE_SIZE: int = 100 * 1024 * 1024  # 100 MB (increased)
    ALLOWED_EXTENSIONS: list = [".pdf", ".jpg", ".jpeg", ".png", ".tiff", ".bmp"]
    
    # ==================== PROCESSING SETTINGS ====================
    BATCH_SIZE: int = 32
    MAX_WORKERS: int = 4
    
    # ==================== CACHING (NEW) ====================
    ENABLE_EMBEDDING_CACHE: bool = True
    CACHE_TTL: int = 3600  # 1 hour
    REDIS_URL: Optional[str] = "redis://localhost:6379"

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
