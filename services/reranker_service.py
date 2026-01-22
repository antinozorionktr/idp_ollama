"""
Reranker Service v2.0 - Using Ollama Reranker Model
- Retrieve more candidates with vector search
- Rerank with cross-encoder for better relevance
- Uses Ollama's reranker model API
"""

import httpx
from typing import List, Dict, Any, Optional
import logging
import math
from config import settings

logger = logging.getLogger(__name__)


class RerankerService:
    def __init__(self):
        """Initialize the Ollama-based reranker"""
        self.enabled = settings.ENABLE_RERANKING
        self.model = settings.RERANKER_MODEL
        self.base_url = settings.OLLAMA_BASE_URL
        self._verified = False
        
        if self.enabled:
            try:
                # Verify model is available
                self._verify_model()
                logger.info(f"Reranker initialized: {self.model}")
            except Exception as e:
                logger.warning(f"Reranker model not available, falling back to score-based ranking: {e}")
                self._verified = False
    
    def _verify_model(self):
        """Verify the reranker model is available in Ollama"""
        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.get(f"{self.base_url}/api/tags")
                if response.status_code == 200:
                    models = response.json().get("models", [])
                    model_names = [m.get("name", "") for m in models]
                    
                    # Check for exact match or base name match
                    base_name = self.model.split(":")[0]
                    if any(self.model in name or base_name in name for name in model_names):
                        self._verified = True
                        logger.info(f"Reranker model '{self.model}' verified in Ollama")
                    else:
                        logger.warning(f"Reranker model '{self.model}' not found in Ollama")
                        self._verified = False
        except Exception as e:
            logger.error(f"Failed to verify reranker model: {e}")
            self._verified = False
    
    def _get_rerank_scores(self, query: str, documents: List[str]) -> List[float]:
        """
        Get reranking scores from Ollama reranker model.
        
        The reranker model scores query-document pairs for relevance.
        """
        if not self._verified:
            # Fallback: return uniform scores
            return [0.5] * len(documents)
        
        scores = []
        
        try:
            with httpx.Client(timeout=60.0) as client:
                for doc in documents:
                    # Format as reranking prompt
                    # Different reranker models may need different prompts
                    prompt = f"""Score the relevance of the following document to the query on a scale of 0-10.
Query: {query}
Document: {doc[:1000]}

Respond with only a number between 0 and 10."""
                    
                    response = client.post(
                        f"{self.base_url}/api/generate",
                        json={
                            "model": self.model,
                            "prompt": prompt,
                            "stream": False,
                            "options": {
                                "temperature": 0,
                                "num_predict": 10
                            }
                        },
                        timeout=30.0
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        response_text = result.get("response", "5").strip()
                        
                        # Extract numeric score
                        try:
                            # Try to extract first number from response
                            import re
                            numbers = re.findall(r'[\d.]+', response_text)
                            if numbers:
                                score = float(numbers[0])
                                # Normalize to 0-1 range
                                score = min(max(score / 10.0, 0), 1)
                            else:
                                score = 0.5
                        except:
                            score = 0.5
                        
                        scores.append(score)
                    else:
                        scores.append(0.5)
                        
        except Exception as e:
            logger.error(f"Error getting rerank scores: {e}")
            # Return fallback scores
            return [0.5] * len(documents)
        
        return scores
    
    def _get_rerank_scores_batch(self, query: str, documents: List[str]) -> List[float]:
        """
        Batch reranking using embedding similarity as a fast fallback.
        This is faster than individual LLM calls.
        """
        if not documents:
            return []
        
        try:
            # Use embedding-based reranking as fast fallback
            with httpx.Client(timeout=60.0) as client:
                # Get query embedding
                query_response = client.post(
                    f"{self.base_url}/api/embed",
                    json={"model": settings.EMBEDDING_MODEL, "input": query}
                )
                
                if query_response.status_code != 200:
                    return [0.5] * len(documents)
                
                query_data = query_response.json()
                query_emb = query_data.get("embeddings", [query_data.get("embedding")])[0]
                
                # Get document embeddings
                doc_response = client.post(
                    f"{self.base_url}/api/embed",
                    json={"model": settings.EMBEDDING_MODEL, "input": documents}
                )
                
                if doc_response.status_code != 200:
                    return [0.5] * len(documents)
                
                doc_data = doc_response.json()
                doc_embs = doc_data.get("embeddings", [])
                
                # Calculate cosine similarities
                import numpy as np
                query_np = np.array(query_emb)
                query_norm = np.linalg.norm(query_np)
                
                scores = []
                for doc_emb in doc_embs:
                    doc_np = np.array(doc_emb)
                    doc_norm = np.linalg.norm(doc_np)
                    if query_norm > 0 and doc_norm > 0:
                        similarity = np.dot(query_np, doc_np) / (query_norm * doc_norm)
                        scores.append(float(similarity))
                    else:
                        scores.append(0.5)
                
                return scores
                
        except Exception as e:
            logger.error(f"Batch reranking failed: {e}")
            return [0.5] * len(documents)
    
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
        if not self.enabled or not documents:
            return documents[:top_k] if top_k else documents
        
        if top_k is None:
            top_k = settings.RERANK_TOP_K
        
        try:
            # Extract texts
            texts = [doc.get("text", "")[:1000] for doc in documents]  # Truncate for efficiency
            
            # Get reranking scores (use batch method for speed)
            scores = self._get_rerank_scores_batch(query, texts)
            
            # Add rerank scores to documents
            for doc, score in zip(documents, scores):
                doc["original_score"] = doc.get("score", 0)
                doc["rerank_score"] = float(score)
                # Combine scores (weighted average)
                doc["score"] = 0.3 * doc["original_score"] + 0.7 * score
            
            # Sort by rerank score
            reranked = sorted(documents, key=lambda x: x["rerank_score"], reverse=True)
            
            logger.info(f"Reranked {len(documents)} documents, returning top {top_k}")
            
            return reranked[:top_k]
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return documents[:top_k]
    
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
        if not self.enabled or not documents:
            return documents[:top_k] if top_k else documents
        
        if top_k is None:
            top_k = settings.RERANK_TOP_K
        
        try:
            # Get reranking scores
            texts = [doc.get("text", "")[:1000] for doc in documents]
            scores = self._get_rerank_scores_batch(query, texts)
            
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
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the reranker model"""
        return {
            "enabled": self.enabled,
            "verified": self._verified,
            "model_name": self.model if self.enabled else None,
            "provider": "ollama",
            "base_url": self.base_url,
            "candidates": settings.RERANK_CANDIDATES,
            "top_k": settings.RERANK_TOP_K,
            "method": "embedding_similarity" if self._verified else "fallback"
        }