"""
Vector Store Service v2.0 - Hybrid Search
- Dense vector search (semantic)
- Sparse BM25 search (keyword)
- Weighted combination for best results
"""

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, 
    Filter, FieldCondition, MatchValue,
    SparseVectorParams, SparseIndexParams,
    NamedVector, NamedSparseVector,
    SparseVector, SearchRequest, FusionQuery,
    Prefetch, Query
)
from typing import List, Dict, Any, Optional, Tuple
import logging
from config import settings
import uuid
import math
from collections import Counter
import re

logger = logging.getLogger(__name__)


class BM25Encoder:
    """Simple BM25 encoder for sparse vectors"""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.vocab = {}
        self.idf = {}
        self.avg_doc_len = 0
        self.doc_count = 0
    
    def fit(self, documents: List[str]):
        """Fit BM25 on document corpus"""
        doc_freqs = Counter()
        total_len = 0
        
        for doc in documents:
            tokens = self._tokenize(doc)
            total_len += len(tokens)
            unique_tokens = set(tokens)
            for token in unique_tokens:
                doc_freqs[token] += 1
        
        self.doc_count = len(documents)
        self.avg_doc_len = total_len / self.doc_count if self.doc_count > 0 else 0
        
        # Build vocabulary and IDF
        for idx, (token, freq) in enumerate(doc_freqs.items()):
            self.vocab[token] = idx
            # IDF with smoothing
            self.idf[token] = math.log((self.doc_count - freq + 0.5) / (freq + 0.5) + 1)
    
    def encode(self, text: str) -> Tuple[List[int], List[float]]:
        """Encode text to sparse vector (indices, values)"""
        tokens = self._tokenize(text)
        doc_len = len(tokens)
        term_freqs = Counter(tokens)
        
        indices = []
        values = []
        
        for token, tf in term_freqs.items():
            if token in self.vocab:
                idx = self.vocab[token]
                idf = self.idf.get(token, 0)
                
                # BM25 score
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / (self.avg_doc_len + 1))
                score = idf * numerator / denominator
                
                indices.append(idx)
                values.append(score)
        
        return indices, values
    
    def encode_query(self, query: str) -> Tuple[List[int], List[float]]:
        """Encode query (simpler weighting)"""
        tokens = self._tokenize(query)
        term_freqs = Counter(tokens)
        
        indices = []
        values = []
        
        for token, tf in term_freqs.items():
            if token in self.vocab:
                idx = self.vocab[token]
                idf = self.idf.get(token, 0)
                score = idf * tf  # Simpler query weighting
                
                indices.append(idx)
                values.append(score)
        
        return indices, values
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens


class VectorStoreService:
    def __init__(self):
        """Initialize Qdrant client with hybrid search support"""
        try:
            if settings.QDRANT_API_KEY:
                self.client = QdrantClient(
                    url=f"{'https' if settings.QDRANT_HTTPS else 'http'}://"
                        f"{settings.QDRANT_HOST}:{settings.QDRANT_PORT}",
                    api_key=settings.QDRANT_API_KEY
                )
            else:
                self.client = QdrantClient(
                    host=settings.QDRANT_HOST,
                    port=settings.QDRANT_PORT
                )
            
            # Hybrid search settings
            self.hybrid_enabled = settings.ENABLE_HYBRID_SEARCH
            self.vector_weight = settings.VECTOR_SEARCH_WEIGHT
            self.bm25_weight = 1 - self.vector_weight
            
            # BM25 encoder for sparse vectors
            self.bm25_encoders = {}  # Per-collection encoders
            
            logger.info(f"Connected to Qdrant at {settings.QDRANT_HOST}:{settings.QDRANT_PORT}")
            logger.info(f"Hybrid search: {'enabled' if self.hybrid_enabled else 'disabled'}")
            
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {str(e)}")
            raise
    
    def create_collection(
        self, 
        collection_name: str, 
        vector_size: int = None
    ) -> bool:
        """Create a new collection with hybrid search support"""
        try:
            if vector_size is None:
                vector_size = settings.EMBEDDING_DIMENSION
            
            # Check if collection exists
            collections = self.client.get_collections().collections
            exists = any(c.name == collection_name for c in collections)
            
            if not exists:
                if self.hybrid_enabled:
                    # Create collection with both dense and sparse vectors
                    self.client.create_collection(
                        collection_name=collection_name,
                        vectors_config={
                            "dense": VectorParams(
                                size=vector_size,
                                distance=Distance.COSINE
                            )
                        },
                        sparse_vectors_config={
                            "sparse": SparseVectorParams(
                                index=SparseIndexParams(on_disk=False)
                            )
                        }
                    )
                    logger.info(f"Created hybrid collection: {collection_name}")
                else:
                    # Dense only
                    self.client.create_collection(
                        collection_name=collection_name,
                        vectors_config=VectorParams(
                            size=vector_size,
                            distance=Distance.COSINE
                        )
                    )
                    logger.info(f"Created collection: {collection_name}")
                
                # Initialize BM25 encoder for this collection
                self.bm25_encoders[collection_name] = BM25Encoder(
                    k1=settings.BM25_K1,
                    b=settings.BM25_B
                )
            else:
                logger.info(f"Collection already exists: {collection_name}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating collection: {str(e)}")
            raise
    
    def add_documents(
        self,
        collection_name: str,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]],
        metadata: Dict[str, Any]
    ) -> List[str]:
        """
        Add documents with both dense and sparse vectors
        """
        try:
            # Ensure collection exists
            self.create_collection(collection_name)
            
            if len(chunks) != len(embeddings):
                raise ValueError("Number of chunks must match number of embeddings")
            
            # Fit BM25 on documents
            texts = [chunk["text"] for chunk in chunks]
            if collection_name not in self.bm25_encoders:
                self.bm25_encoders[collection_name] = BM25Encoder(
                    k1=settings.BM25_K1,
                    b=settings.BM25_B
                )
            self.bm25_encoders[collection_name].fit(texts)
            
            # Create points
            points = []
            point_ids = []
            
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                point_id = str(uuid.uuid4())
                point_ids.append(point_id)
                
                # Combine chunk metadata with document metadata
                payload = {
                    "text": chunk["text"],
                    "chunk_id": chunk.get("chunk_id", i),
                    "chunk_type": chunk.get("type", "text"),
                    "page": chunk.get("page", 1),
                    **chunk.get("metadata", {}),
                    **metadata
                }
                
                if "bbox" in chunk:
                    payload["bbox"] = chunk["bbox"]
                
                if self.hybrid_enabled:
                    # Generate sparse vector
                    indices, values = self.bm25_encoders[collection_name].encode(chunk["text"])
                    
                    points.append(
                        PointStruct(
                            id=point_id,
                            vector={
                                "dense": embedding,
                                "sparse": SparseVector(
                                    indices=indices,
                                    values=values
                                )
                            },
                            payload=payload
                        )
                    )
                else:
                    points.append(
                        PointStruct(
                            id=point_id,
                            vector=embedding,
                            payload=payload
                        )
                    )
            
            # Batch upload
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                self.client.upsert(
                    collection_name=collection_name,
                    points=batch
                )
            
            logger.info(f"Added {len(points)} points to collection {collection_name}")
            return point_ids
            
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            raise
    
    def search(
        self,
        collection_name: str,
        query_embedding: List[float],
        query_text: str = None,
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search combining dense and sparse vectors
        """
        try:
            # Build filter
            query_filter = None
            if filter_dict:
                conditions = []
                for key, value in filter_dict.items():
                    conditions.append(
                        FieldCondition(
                            key=key,
                            match=MatchValue(value=value)
                        )
                    )
                query_filter = Filter(must=conditions)
            
            if self.hybrid_enabled and query_text:
                # Hybrid search with RRF (Reciprocal Rank Fusion)
                results = self._hybrid_search(
                    collection_name=collection_name,
                    query_embedding=query_embedding,
                    query_text=query_text,
                    top_k=top_k,
                    query_filter=query_filter
                )
            else:
                # Dense-only search
                results = self.client.query_points(
                    collection_name=collection_name,
                    query=query_embedding,
                    limit=top_k,
                    query_filter=query_filter,
                    score_threshold=score_threshold
                ).points
            
            # Format results
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "id": result.id,
                    "score": result.score,
                    "text": result.payload.get("text", ""),
                    "metadata": {
                        k: v for k, v in result.payload.items() 
                        if k != "text"
                    }
                })
            
            logger.info(f"Found {len(formatted_results)} results for query")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching: {str(e)}")
            raise
    
    def _hybrid_search(
        self,
        collection_name: str,
        query_embedding: List[float],
        query_text: str,
        top_k: int,
        query_filter: Optional[Filter] = None
    ) -> List:
        """
        Perform hybrid search with RRF fusion
        """
        # Get sparse query vector
        if collection_name in self.bm25_encoders:
            indices, values = self.bm25_encoders[collection_name].encode_query(query_text)
        else:
            indices, values = [], []
        
        # Hybrid search with prefetch and fusion
        try:
            results = self.client.query_points(
                collection_name=collection_name,
                prefetch=[
                    Prefetch(
                        query=query_embedding,
                        using="dense",
                        limit=top_k * 2
                    ),
                    Prefetch(
                        query=SparseVector(indices=indices, values=values),
                        using="sparse",
                        limit=top_k * 2
                    )
                ],
                query=FusionQuery(fusion="rrf"),  # Reciprocal Rank Fusion
                limit=top_k,
                query_filter=query_filter
            ).points
            
            return results
            
        except Exception as e:
            logger.warning(f"Hybrid search failed, falling back to dense: {e}")
            # Fallback to dense-only
            return self.client.query_points(
                collection_name=collection_name,
                query=query_embedding,
                using="dense" if self.hybrid_enabled else None,
                limit=top_k,
                query_filter=query_filter
            ).points
    
    def search_with_candidates(
        self,
        collection_name: str,
        query_embedding: List[float],
        query_text: str = None,
        candidates: int = 20,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search and return more candidates for reranking
        """
        return self.search(
            collection_name=collection_name,
            query_embedding=query_embedding,
            query_text=query_text,
            top_k=candidates,
            filter_dict=filter_dict
        )
    
    def delete_document(
        self, 
        collection_name: str, 
        document_id: str
    ) -> int:
        """Delete all chunks belonging to a document"""
        try:
            self.client.delete(
                collection_name=collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=document_id)
                        )
                    ]
                )
            )
            
            logger.info(f"Deleted document {document_id} from {collection_name}")
            return 1
            
        except Exception as e:
            logger.error(f"Error deleting document: {str(e)}")
            raise
    
    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """Get information about a collection"""
        try:
            info = self.client.get_collection(collection_name)
            return {
                "name": collection_name,
                "vectors_count": info.points_count,
                "segments_count": info.segments_count,
                "hybrid_enabled": self.hybrid_enabled,
                "config": {
                    "vector_size": info.config.params.vectors.size if hasattr(info.config.params.vectors, 'size') else "hybrid",
                    "distance": "COSINE"
                }
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {str(e)}")
            raise
    
    def list_collections(self) -> List[str]:
        """List all collections"""
        try:
            collections = self.client.get_collections().collections
            return [c.name for c in collections]
        except Exception as e:
            logger.error(f"Error listing collections: {str(e)}")
            raise
    
    def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection"""
        try:
            self.client.delete_collection(collection_name)
            if collection_name in self.bm25_encoders:
                del self.bm25_encoders[collection_name]
            logger.info(f"Deleted collection: {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error deleting collection: {str(e)}")
            raise
    
    def health_check(self) -> bool:
        """Check if Qdrant is accessible"""
        try:
            self.client.get_collections()
            return True
        except Exception as e:
            logger.error(f"Qdrant health check failed: {str(e)}")
            raise
