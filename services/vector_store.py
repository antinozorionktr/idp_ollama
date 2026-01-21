"""
Vector Store Service
Manages document storage and retrieval using Qdrant

Integrates with Tier 2 (Embedding) for indexing and search.
"""

import logging
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    MatchAny,
    Range,
    SearchRequest,
    UpdateStatus
)

from config import settings

logger = logging.getLogger(__name__)


class VectorStoreService:
    """
    Vector store service using Qdrant for document indexing and retrieval.
    """
    
    def __init__(self):
        """Initialize Qdrant client"""
        try:
            if settings.QDRANT_API_KEY:
                # Cloud Qdrant
                protocol = "https" if settings.QDRANT_HTTPS else "http"
                self.client = QdrantClient(
                    url=f"{protocol}://{settings.QDRANT_HOST}:{settings.QDRANT_PORT}",
                    api_key=settings.QDRANT_API_KEY
                )
            else:
                # Local Qdrant
                self.client = QdrantClient(
                    host=settings.QDRANT_HOST,
                    port=settings.QDRANT_PORT
                )
            
            logger.info(f"Connected to Qdrant at {settings.QDRANT_HOST}:{settings.QDRANT_PORT}")
            
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise
    
    def health_check(self) -> bool:
        """Check if Qdrant is accessible"""
        try:
            self.client.get_collections()
            return True
        except Exception as e:
            logger.error(f"Qdrant health check failed: {e}")
            return False
    
    def ensure_collection(
        self,
        collection_name: str,
        vector_size: int = None
    ) -> bool:
        """
        Ensure a collection exists, creating it if necessary.
        
        Args:
            collection_name: Name of the collection
            vector_size: Vector dimension (default from settings)
            
        Returns:
            True if collection exists or was created
        """
        if vector_size is None:
            vector_size = settings.EMBEDDING_DIMENSION
        
        try:
            collections = self.client.get_collections().collections
            exists = any(c.name == collection_name for c in collections)
            
            if not exists:
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created collection: {collection_name}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error ensuring collection: {e}")
            raise
    
    def add_documents(
        self,
        collection_name: str,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]],
        document_metadata: Dict[str, Any]
    ) -> List[str]:
        """
        Add document chunks to the vector store.
        
        Args:
            collection_name: Target collection
            chunks: List of chunk dicts with text and metadata
            embeddings: Corresponding embedding vectors
            document_metadata: Document-level metadata
            
        Returns:
            List of point IDs
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunks ({len(chunks)}) and embeddings ({len(embeddings)}) must match"
            )
        
        self.ensure_collection(collection_name)
        
        points = []
        point_ids = []
        
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            point_id = str(uuid.uuid4())
            point_ids.append(point_id)
            
            # Combine chunk and document metadata
            payload = {
                "text": chunk.get("text", ""),
                "chunk_id": chunk.get("chunk_id", str(i)),
                "content_type": chunk.get("content_type", "text"),
                "page_number": chunk.get("page_number", 1),
                "indexed_at": datetime.utcnow().isoformat(),
                **chunk.get("metadata", {}),
                **document_metadata
            }
            
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
        
        logger.info(f"Added {len(points)} points to collection '{collection_name}'")
        return point_ids
    
    def search(
        self,
        collection_name: str,
        query_embedding: List[float],
        top_k: int = 5,
        filter_conditions: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents.
        
        Args:
            collection_name: Collection to search
            query_embedding: Query vector
            top_k: Number of results
            filter_conditions: Metadata filters
            score_threshold: Minimum similarity score
            
        Returns:
            List of results with text, metadata, and score
        """
        # Build filter
        query_filter = None
        if filter_conditions:
            must_conditions = []
            for key, value in filter_conditions.items():
                if isinstance(value, list):
                    must_conditions.append(
                        FieldCondition(key=key, match=MatchAny(any=value))
                    )
                else:
                    must_conditions.append(
                        FieldCondition(key=key, match=MatchValue(value=value))
                    )
            query_filter = Filter(must=must_conditions)
        
        try:
            results = self.client.query_points(
                collection_name=collection_name,
                query=query_embedding,
                limit=top_k,
                query_filter=query_filter,
                score_threshold=score_threshold or settings.SIMILARITY_THRESHOLD
            ).points
            
            formatted = []
            for result in results:
                formatted.append({
                    "id": result.id,
                    "score": result.score,
                    "text": result.payload.get("text", ""),
                    "page_number": result.payload.get("page_number"),
                    "document_id": result.payload.get("document_id"),
                    "filename": result.payload.get("filename"),
                    "content_type": result.payload.get("content_type"),
                    "metadata": {
                        k: v for k, v in result.payload.items()
                        if k not in ["text"]
                    }
                })
            
            logger.info(f"Found {len(formatted)} results in '{collection_name}'")
            return formatted
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            raise
    
    def delete_document(
        self,
        collection_name: str,
        document_id: str
    ) -> int:
        """
        Delete all chunks belonging to a document.
        
        Args:
            collection_name: Collection name
            document_id: Document ID to delete
            
        Returns:
            Number of deleted points (estimated)
        """
        try:
            # First count the points (for return value)
            results = self.client.scroll(
                collection_name=collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=document_id)
                        )
                    ]
                ),
                limit=1000
            )
            count = len(results[0])
            
            # Delete
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
            
            logger.info(f"Deleted {count} chunks for document {document_id}")
            return count
            
        except Exception as e:
            logger.error(f"Delete error: {e}")
            raise
    
    def list_collections(self) -> List[Dict[str, Any]]:
        """
        List all collections with their info.
        
        Returns:
            List of collection info dicts
        """
        try:
            collections = self.client.get_collections().collections
            
            result = []
            for c in collections:
                info = self.client.get_collection(c.name)
                result.append({
                    "name": c.name,
                    "vectors_count": info.points_count,
                    "status": info.status.name if hasattr(info, 'status') else "unknown"
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Error listing collections: {e}")
            raise
    
    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a collection.
        
        Args:
            collection_name: Collection name
            
        Returns:
            Collection info dict
        """
        try:
            info = self.client.get_collection(collection_name)
            
            return {
                "name": collection_name,
                "vectors_count": info.points_count,
                "segments_count": info.segments_count,
                "status": info.status.name if hasattr(info, 'status') else "unknown",
                "config": {
                    "vector_size": info.config.params.vectors.size,
                    "distance": info.config.params.vectors.distance.name
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            raise
    
    def delete_collection(self, collection_name: str) -> bool:
        """
        Delete a collection.
        
        Args:
            collection_name: Collection to delete
            
        Returns:
            True if deleted successfully
        """
        try:
            self.client.delete_collection(collection_name)
            logger.info(f"Deleted collection: {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            raise
    
    def get_document_chunks(
        self,
        collection_name: str,
        document_id: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get all chunks for a specific document.
        
        Args:
            collection_name: Collection name
            document_id: Document ID
            limit: Maximum chunks to return
            
        Returns:
            List of chunk data
        """
        try:
            results, _ = self.client.scroll(
                collection_name=collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=document_id)
                        )
                    ]
                ),
                limit=limit,
                with_payload=True,
                with_vectors=False
            )
            
            chunks = []
            for point in results:
                chunks.append({
                    "id": point.id,
                    "text": point.payload.get("text", ""),
                    "page_number": point.payload.get("page_number"),
                    "content_type": point.payload.get("content_type"),
                    "chunk_id": point.payload.get("chunk_id"),
                    "metadata": point.payload
                })
            
            # Sort by page and chunk order
            chunks.sort(key=lambda x: (
                x.get("page_number", 0),
                int(x.get("chunk_id", 0)) if str(x.get("chunk_id", "0")).isdigit() else 0
            ))
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error getting document chunks: {e}")
            raise
    
    def get_unique_documents(
        self,
        collection_name: str
    ) -> List[Dict[str, Any]]:
        """
        Get list of unique documents in a collection.
        
        Args:
            collection_name: Collection name
            
        Returns:
            List of unique document info
        """
        try:
            # Scroll through all points to get unique document IDs
            documents = {}
            offset = None
            
            while True:
                results, offset = self.client.scroll(
                    collection_name=collection_name,
                    limit=100,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False
                )
                
                for point in results:
                    doc_id = point.payload.get("document_id")
                    if doc_id and doc_id not in documents:
                        documents[doc_id] = {
                            "document_id": doc_id,
                            "filename": point.payload.get("filename"),
                            "num_pages": point.payload.get("num_pages"),
                            "indexed_at": point.payload.get("indexed_at"),
                            "chunk_count": 0
                        }
                    if doc_id:
                        documents[doc_id]["chunk_count"] += 1
                
                if offset is None:
                    break
            
            return list(documents.values())
            
        except Exception as e:
            logger.error(f"Error getting unique documents: {e}")
            raise
