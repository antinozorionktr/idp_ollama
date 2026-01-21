"""
Tier 2: Embedding Service (The "Memory")
Uses nomic-embed-text for lightweight, long-context optimized embeddings

Key features:
- Optimized for retrieval tasks
- Handles long documents efficiently
- Runs locally via Ollama
- Low VRAM footprint
"""

import logging
from typing import List, Optional
import httpx
import asyncio
from datetime import datetime

from config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Embedding service using nomic-embed-text via Ollama.
    Generates vector embeddings for document chunks.
    """
    
    def __init__(self):
        self.base_url = settings.OLLAMA_BASE_URL
        self.model = settings.EMBEDDING_MODEL
        self.dimension = settings.EMBEDDING_DIMENSION
        self.batch_size = settings.EMBEDDING_BATCH_SIZE
        self._client: Optional[httpx.AsyncClient] = None
        logger.info(f"EmbeddingService initialized with model: {self.model}")
    
    @property
    def client(self) -> httpx.AsyncClient:
        """Lazy initialization of async HTTP client"""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(60.0)
            )
        return self._client
    
    async def close(self):
        """Close the HTTP client"""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
    
    async def check_model_available(self) -> bool:
        """Check if the embedding model is available"""
        try:
            response = await self.client.get("/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name", "") for m in models]
                return any(
                    self.model in name or name.startswith(self.model.split(":")[0])
                    for name in model_names
                )
            return False
        except Exception as e:
            logger.error(f"Error checking embedding model: {e}")
            return False
    
    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            return [0.0] * self.dimension
        
        payload = {
            "model": self.model,
            "prompt": text
        }
        
        try:
            response = await self.client.post(
                "/api/embeddings",
                json=payload
            )
            response.raise_for_status()
            
            result = response.json()
            embedding = result.get("embedding", [])
            
            if len(embedding) != self.dimension:
                logger.warning(
                    f"Unexpected embedding dimension: {len(embedding)} "
                    f"(expected {self.dimension})"
                )
            
            return embedding
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise
    
    async def generate_embeddings(
        self,
        texts: List[str],
        show_progress: bool = False
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Processes in batches for efficiency.
        
        Args:
            texts: List of texts to embed
            show_progress: Whether to log progress
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        start_time = datetime.utcnow()
        all_embeddings = []
        total = len(texts)
        
        # Process in batches
        for i in range(0, total, self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            # Generate embeddings concurrently within batch
            tasks = [self.generate_embedding(text) for text in batch]
            batch_embeddings = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle any errors
            for j, emb in enumerate(batch_embeddings):
                if isinstance(emb, Exception):
                    logger.error(f"Embedding error for text {i+j}: {emb}")
                    # Use zero vector as fallback
                    all_embeddings.append([0.0] * self.dimension)
                else:
                    all_embeddings.append(emb)
            
            if show_progress:
                processed = min(i + self.batch_size, total)
                logger.info(f"Embedded {processed}/{total} texts")
        
        duration = (datetime.utcnow() - start_time).total_seconds()
        logger.info(
            f"Generated {len(all_embeddings)} embeddings in {duration:.2f}s "
            f"({len(all_embeddings)/duration:.1f} texts/sec)"
        )
        
        return all_embeddings
    
    async def generate_query_embedding(self, query: str) -> List[float]:
        """
        Generate embedding for a search query.
        
        Note: nomic-embed-text uses the same model for documents and queries,
        but we keep this separate method for API consistency and potential
        future query-specific preprocessing.
        
        Args:
            query: Search query text
            
        Returns:
            Query embedding vector
        """
        # nomic-embed-text benefits from a search prefix for queries
        # This helps differentiate query intent from document content
        prefixed_query = f"search_query: {query}"
        return await self.generate_embedding(prefixed_query)
    
    async def generate_document_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a document chunk.
        
        Args:
            text: Document text to embed
            
        Returns:
            Document embedding vector
        """
        # nomic-embed-text benefits from a document prefix
        prefixed_text = f"search_document: {text}"
        return await self.generate_embedding(prefixed_text)
    
    async def generate_document_embeddings(
        self,
        texts: List[str],
        show_progress: bool = False
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple document chunks with proper prefix.
        
        Args:
            texts: List of document texts
            show_progress: Whether to log progress
            
        Returns:
            List of embedding vectors
        """
        prefixed_texts = [f"search_document: {text}" for text in texts]
        return await self.generate_embeddings(prefixed_texts, show_progress)
    
    def cosine_similarity(
        self,
        embedding1: List[float],
        embedding2: List[float]
    ) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score (0-1)
        """
        import math
        
        if len(embedding1) != len(embedding2):
            raise ValueError("Embeddings must have same dimension")
        
        dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
        norm1 = math.sqrt(sum(a * a for a in embedding1))
        norm2 = math.sqrt(sum(b * b for b in embedding2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    async def find_most_similar(
        self,
        query: str,
        candidates: List[str],
        top_k: int = 5
    ) -> List[dict]:
        """
        Find most similar texts to a query from a list of candidates.
        
        This is useful for quick similarity search without a vector database.
        
        Args:
            query: Query text
            candidates: List of candidate texts
            top_k: Number of results to return
            
        Returns:
            List of dicts with 'text', 'score', 'index'
        """
        if not candidates:
            return []
        
        # Generate embeddings
        query_emb = await self.generate_query_embedding(query)
        candidate_embs = await self.generate_document_embeddings(candidates)
        
        # Calculate similarities
        similarities = []
        for i, (text, emb) in enumerate(zip(candidates, candidate_embs)):
            score = self.cosine_similarity(query_emb, emb)
            similarities.append({
                "text": text,
                "score": score,
                "index": i
            })
        
        # Sort by score descending
        similarities.sort(key=lambda x: x["score"], reverse=True)
        
        return similarities[:top_k]
