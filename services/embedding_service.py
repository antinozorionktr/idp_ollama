"""
Embedding Service using bge-large-en-v1.5
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Any
import logging
import torch
from config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self):
        """Initialize BGE embedding model"""
        try:
            # Load BGE model
            self.model = SentenceTransformer(
                settings.EMBEDDING_MODEL,
                device=settings.EMBEDDING_DEVICE
            )
            
            # Verify embedding dimension
            test_embedding = self.model.encode(["test"], show_progress_bar=False)
            actual_dim = test_embedding.shape[1]
            
            if actual_dim != settings.EMBEDDING_DIMENSION:
                logger.warning(
                    f"Embedding dimension mismatch: expected {settings.EMBEDDING_DIMENSION}, "
                    f"got {actual_dim}. Updating setting."
                )
                settings.EMBEDDING_DIMENSION = actual_dim
            
            logger.info(
                f"Embedding model initialized: {settings.EMBEDDING_MODEL} "
                f"(dimension: {settings.EMBEDDING_DIMENSION})"
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {str(e)}")
            raise
    
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
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        # Clean texts
        texts = [self._clean_text(text) for text in texts]
        
        # Set batch size
        if batch_size is None:
            batch_size = settings.BATCH_SIZE
        
        try:
            # Generate embeddings
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=True  # Normalize for cosine similarity
            )
            
            # Convert to list of lists
            embeddings_list = embeddings.tolist()
            
            logger.info(f"Generated {len(embeddings_list)} embeddings")
            return embeddings_list
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise
    
    def generate_query_embedding(self, query: str) -> List[float]:
        """
        Generate embedding for a single query
        
        BGE models use special instructions for queries vs passages
        """
        # Add BGE query instruction
        query_with_instruction = f"Represent this sentence for searching relevant passages: {query}"
        
        embedding = self.model.encode(
            [query_with_instruction],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        return embedding[0].tolist()
    
    def _clean_text(self, text: str) -> str:
        """Clean and preprocess text"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = " ".join(text.split())
        
        # Truncate if too long (BGE has max length)
        max_length = 512  # BGE max token length
        words = text.split()
        if len(words) > max_length:
            text = " ".join(words[:max_length])
        
        return text
    
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
        return settings.EMBEDDING_DIMENSION
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the embedding model"""
        return {
            "model_name": settings.EMBEDDING_MODEL,
            "dimension": settings.EMBEDDING_DIMENSION,
            "device": settings.EMBEDDING_DEVICE,
            "max_sequence_length": self.model.max_seq_length
        }
