"""
Embedding Service using Ollama's local embedding models
Supports: nomic-embed-text, mxbai-embed-large, all-minilm, etc.
"""

import httpx
import numpy as np
from typing import List, Dict, Any, Optional
import logging
import asyncio
from config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self):
        """Initialize Ollama embedding client"""
        self.base_url = settings.OLLAMA_BASE_URL
        self.model = settings.EMBEDDING_MODEL
        self.dimension = settings.EMBEDDING_DIMENSION
        self._client = None
        self._async_client = None
        
        # Verify connection and get actual embedding dimension
        self._verify_connection()
        
        logger.info(
            f"Embedding service initialized with Ollama model: {self.model} "
            f"(dimension: {self.dimension})"
        )
    
    def _verify_connection(self):
        """Verify Ollama connection and embedding model availability"""
        try:
            # Test embedding to verify model and get dimension
            test_response = self._embed_sync(["test"])
            if test_response:
                actual_dim = len(test_response[0])
                if actual_dim != self.dimension:
                    logger.warning(
                        f"Embedding dimension mismatch: configured {self.dimension}, "
                        f"actual {actual_dim}. Updating setting."
                    )
                    self.dimension = actual_dim
                    settings.EMBEDDING_DIMENSION = actual_dim
                logger.info(f"Ollama embedding model '{self.model}' verified successfully")
            else:
                raise Exception("Empty response from embedding model")
        except Exception as e:
            logger.error(f"Failed to verify Ollama embedding model: {e}")
            raise RuntimeError(
                f"Cannot connect to Ollama embedding service at {self.base_url}. "
                f"Make sure Ollama is running and model '{self.model}' is available. "
                f"Run: ollama pull {self.model}"
            )
    
    def _embed_sync(self, texts: List[str]) -> List[List[float]]:
        """Synchronous embedding generation"""
        embeddings = []
        
        with httpx.Client(timeout=60.0) as client:
            for text in texts:
                response = client.post(
                    f"{self.base_url}/api/embed",
                    json={
                        "model": self.model,
                        "input": text
                    }
                )
                response.raise_for_status()
                data = response.json()
                
                # Handle both single and batch response formats
                if "embeddings" in data:
                    embeddings.extend(data["embeddings"])
                elif "embedding" in data:
                    embeddings.append(data["embedding"])
                else:
                    raise ValueError(f"Unexpected response format: {data.keys()}")
        
        return embeddings
    
    async def _embed_async(self, texts: List[str]) -> List[List[float]]:
        """Asynchronous embedding generation"""
        embeddings = []
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            for text in texts:
                response = await client.post(
                    f"{self.base_url}/api/embed",
                    json={
                        "model": self.model,
                        "input": text
                    }
                )
                response.raise_for_status()
                data = response.json()
                
                if "embeddings" in data:
                    embeddings.extend(data["embeddings"])
                elif "embedding" in data:
                    embeddings.append(data["embedding"])
                else:
                    raise ValueError(f"Unexpected response format: {data.keys()}")
        
        return embeddings
    
    def _embed_batch_sync(self, texts: List[str]) -> List[List[float]]:
        """Batch embedding using Ollama's batch endpoint"""
        with httpx.Client(timeout=120.0) as client:
            response = client.post(
                f"{self.base_url}/api/embed",
                json={
                    "model": self.model,
                    "input": texts  # Send all texts at once
                }
            )
            response.raise_for_status()
            data = response.json()
            
            if "embeddings" in data:
                return data["embeddings"]
            elif "embedding" in data:
                return [data["embedding"]]
            else:
                raise ValueError(f"Unexpected response format: {data.keys()}")
    
    def generate_embeddings(
        self, 
        texts: List[str],
        batch_size: int = None,
        show_progress: bool = False
    ) -> List[List[float]]:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of text strings to embed
            batch_size: Batch size for encoding (used for chunking large requests)
            show_progress: Whether to show progress (logged)
            
        Returns:
            List of embedding vectors (normalized)
        """
        if not texts:
            return []
        
        # Clean texts
        texts = [self._clean_text(text) for text in texts]
        
        if batch_size is None:
            batch_size = settings.BATCH_SIZE
        
        try:
            all_embeddings = []
            
            # Process in batches
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                if show_progress:
                    logger.info(f"Processing batch {i // batch_size + 1}/{(len(texts) + batch_size - 1) // batch_size}")
                
                # Use batch endpoint
                batch_embeddings = self._embed_batch_sync(batch)
                all_embeddings.extend(batch_embeddings)
            
            # Normalize embeddings for cosine similarity
            normalized = self._normalize_embeddings(all_embeddings)
            
            logger.info(f"Generated {len(normalized)} embeddings")
            return normalized
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise
    
    def generate_query_embedding(self, query: str) -> List[float]:
        """
        Generate embedding for a single query
        
        For models like nomic-embed-text, you can add search instructions
        """
        # Clean the query
        query = self._clean_text(query)
        
        # For nomic-embed-text, add search task prefix for better results
        if "nomic" in self.model.lower():
            query = f"search_query: {query}"
        
        try:
            embeddings = self._embed_batch_sync([query])
            if embeddings:
                # Normalize the embedding
                normalized = self._normalize_embeddings(embeddings)
                return normalized[0]
            raise ValueError("Empty embedding returned")
        except Exception as e:
            logger.error(f"Error generating query embedding: {e}")
            raise
    
    def _clean_text(self, text: str) -> str:
        """Clean and preprocess text"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = " ".join(text.split())
        
        # Truncate if too long (most embedding models have ~8192 token limit)
        max_chars = 8000  # Approximate character limit
        if len(text) > max_chars:
            text = text[:max_chars]
        
        return text
    
    def _normalize_embeddings(self, embeddings: List[List[float]]) -> List[List[float]]:
        """Normalize embeddings to unit vectors for cosine similarity"""
        normalized = []
        for emb in embeddings:
            arr = np.array(emb)
            norm = np.linalg.norm(arr)
            if norm > 0:
                normalized.append((arr / norm).tolist())
            else:
                normalized.append(emb)
        return normalized
    
    def compute_similarity(
        self, 
        embedding1: List[float], 
        embedding2: List[float]
    ) -> float:
        """
        Compute cosine similarity between two embeddings
        
        Since embeddings are normalized, dot product = cosine similarity
        """
        return float(np.dot(embedding1, embedding2))
    
    def batch_similarity(
        self, 
        query_embedding: List[float], 
        doc_embeddings: List[List[float]]
    ) -> List[float]:
        """
        Compute similarity between a query and multiple documents
        """
        query_np = np.array(query_embedding)
        docs_np = np.array(doc_embeddings)
        
        # Dot product for normalized vectors
        similarities = np.dot(docs_np, query_np)
        
        return similarities.tolist()
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings"""
        return self.dimension
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the embedding model"""
        return {
            "model_name": self.model,
            "dimension": self.dimension,
            "provider": "ollama",
            "base_url": self.base_url
        }


# Optional: Async version for high-throughput scenarios
class AsyncEmbeddingService(EmbeddingService):
    """Async version of EmbeddingService for use in async contexts"""
    
    async def generate_embeddings_async(
        self, 
        texts: List[str],
        batch_size: int = None
    ) -> List[List[float]]:
        """Async version of generate_embeddings"""
        if not texts:
            return []
        
        texts = [self._clean_text(text) for text in texts]
        
        if batch_size is None:
            batch_size = settings.BATCH_SIZE
        
        all_embeddings = []
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                response = await client.post(
                    f"{self.base_url}/api/embed",
                    json={
                        "model": self.model,
                        "input": batch
                    }
                )
                response.raise_for_status()
                data = response.json()
                
                if "embeddings" in data:
                    all_embeddings.extend(data["embeddings"])
                elif "embedding" in data:
                    all_embeddings.append(data["embedding"])
        
        return self._normalize_embeddings(all_embeddings)
