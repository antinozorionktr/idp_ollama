"""
Reranker Service v2.0 - Cross-Encoder Reranking
- Retrieve more candidates with vector search
- Rerank with cross-encoder for better relevance
- Dramatically improves retrieval accuracy
"""

from sentence_transformers import CrossEncoder
from typing import List, Dict, Any, Tuple
import logging
import torch
from config import settings

logger = logging.getLogger(__name__)


class RerankerService:
    def __init__(self):
        """Initialize the cross-encoder reranker model"""
        self.enabled = settings.ENABLE_RERANKING
        self.model = None
        
        if self.enabled:
            try:
                # Load cross-encoder model
                self.model = CrossEncoder(
                    settings.RERANKER_MODEL,
                    max_length=512,
                    device=settings.RERANKER_DEVICE
                )
                
                # Warm up the model
                _ = self.model.predict([("test query", "test document")])
                
                logger.info(f"Reranker initialized: {settings.RERANKER_MODEL}")
                
            except Exception as e:
                logger.error(f"Failed to initialize reranker: {e}")
                self.enabled = False
    
    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = None
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents based on relevance to query
        
        Args:
            query: Search query
            documents: List of documents with 'text' field
            top_k: Number of top results to return (default from settings)
            
        Returns:
            Reranked documents with updated scores
        """
        if not self.enabled or not self.model or not documents:
            return documents[:top_k] if top_k else documents
        
        if top_k is None:
            top_k = settings.RERANK_TOP_K
        
        try:
            # Prepare query-document pairs
            pairs = [(query, doc.get("text", "")) for doc in documents]
            
            # Get reranking scores
            scores = self.model.predict(pairs, show_progress_bar=False)
            
            # Add rerank scores to documents
            for doc, score in zip(documents, scores):
                doc["original_score"] = doc.get("score", 0)
                doc["rerank_score"] = float(score)
                # Combine scores (weighted average)
                doc["score"] = 0.3 * doc["original_score"] + 0.7 * self._normalize_score(score)
            
            # Sort by rerank score
            reranked = sorted(documents, key=lambda x: x["rerank_score"], reverse=True)
            
            logger.info(f"Reranked {len(documents)} documents, returning top {top_k}")
            
            return reranked[:top_k]
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return documents[:top_k]
    
    def _normalize_score(self, score: float) -> float:
        """Normalize reranker score to 0-1 range using sigmoid"""
        import math
        return 1 / (1 + math.exp(-score))
    
    def rerank_with_diversity(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = None,
        diversity_weight: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Rerank with MMR (Maximal Marginal Relevance) for diversity
        
        Balances relevance with diversity to avoid redundant results
        """
        if not self.enabled or not self.model or not documents:
            return documents[:top_k] if top_k else documents
        
        if top_k is None:
            top_k = settings.RERANK_TOP_K
        
        try:
            # Get reranking scores
            pairs = [(query, doc.get("text", "")) for doc in documents]
            scores = self.model.predict(pairs, show_progress_bar=False)
            
            for doc, score in zip(documents, scores):
                doc["rerank_score"] = float(score)
            
            # MMR selection
            selected = []
            candidates = list(range(len(documents)))
            
            while len(selected) < top_k and candidates:
                best_idx = None
                best_mmr = float('-inf')
                
                for idx in candidates:
                    relevance = documents[idx]["rerank_score"]
                    
                    # Calculate max similarity to already selected
                    if selected:
                        max_sim = max(
                            self._text_similarity(
                                documents[idx]["text"],
                                documents[sel_idx]["text"]
                            )
                            for sel_idx in selected
                        )
                    else:
                        max_sim = 0
                    
                    # MMR score
                    mmr = (1 - diversity_weight) * relevance - diversity_weight * max_sim
                    
                    if mmr > best_mmr:
                        best_mmr = mmr
                        best_idx = idx
                
                if best_idx is not None:
                    selected.append(best_idx)
                    candidates.remove(best_idx)
            
            result = [documents[idx] for idx in selected]
            logger.info(f"MMR reranking: selected {len(result)} diverse documents")
            
            return result
            
        except Exception as e:
            logger.error(f"MMR reranking failed: {e}")
            return documents[:top_k]
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Simple Jaccard similarity for diversity calculation"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def batch_rerank(
        self,
        queries: List[str],
        documents_list: List[List[Dict[str, Any]]],
        top_k: int = None
    ) -> List[List[Dict[str, Any]]]:
        """
        Batch reranking for multiple queries
        
        More efficient than calling rerank() multiple times
        """
        if not self.enabled or not self.model:
            return [docs[:top_k] if top_k else docs for docs in documents_list]
        
        if top_k is None:
            top_k = settings.RERANK_TOP_K
        
        results = []
        
        # Flatten all pairs for batch processing
        all_pairs = []
        pair_indices = []  # Track which query-doc list each pair belongs to
        
        for q_idx, (query, documents) in enumerate(zip(queries, documents_list)):
            for doc in documents:
                all_pairs.append((query, doc.get("text", "")))
                pair_indices.append(q_idx)
        
        if not all_pairs:
            return [[] for _ in queries]
        
        try:
            # Batch predict
            all_scores = self.model.predict(all_pairs, show_progress_bar=False)
            
            # Reconstruct results
            score_idx = 0
            for q_idx, documents in enumerate(documents_list):
                for doc in documents:
                    doc["rerank_score"] = float(all_scores[score_idx])
                    score_idx += 1
                
                # Sort and take top_k
                sorted_docs = sorted(
                    documents, 
                    key=lambda x: x.get("rerank_score", 0), 
                    reverse=True
                )
                results.append(sorted_docs[:top_k])
            
            return results
            
        except Exception as e:
            logger.error(f"Batch reranking failed: {e}")
            return [docs[:top_k] if top_k else docs for docs in documents_list]
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the reranker model"""
        return {
            "enabled": self.enabled,
            "model_name": settings.RERANKER_MODEL if self.enabled else None,
            "device": settings.RERANKER_DEVICE if self.enabled else None,
            "candidates": settings.RERANK_CANDIDATES,
            "top_k": settings.RERANK_TOP_K
        }
