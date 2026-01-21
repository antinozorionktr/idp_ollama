"""
IDP System v2 Services
Three-Tier Architecture:
- Tier 1: Vision Service (qwen2.5-vl) - Document understanding
- Tier 2: Embedding Service (nomic-embed-text) - Vector embeddings
- Tier 3: Reasoning Service (phi-4) - QA and synthesis
"""

from .vision_service import VisionService
from .embedding_service import EmbeddingService
from .reasoning_service import ReasoningService
from .vector_store import VectorStoreService
from .chunking_service import ChunkingService

__all__ = [
    "VisionService",
    "EmbeddingService",
    "ReasoningService",
    "VectorStoreService",
    "ChunkingService"
]
