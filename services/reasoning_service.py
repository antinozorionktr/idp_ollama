"""
Tier 3: Reasoning Service (The "Brain")
Uses phi-4:14b or gemma3:12b for high-logic QA and synthesis

Key features:
- Superior instruction-following capabilities
- Context-aware reasoning over retrieved chunks
- Structured answer generation
- Fast inference on 16GB VRAM
"""

import json
import logging
from typing import Dict, Any, Optional, List
import httpx
from datetime import datetime

from config import settings

logger = logging.getLogger(__name__)


class ReasoningService:
    """
    Reasoning service for RAG-based question answering.
    Uses a high-capability local LLM for synthesis and reasoning.
    """
    
    def __init__(self):
        self.base_url = settings.OLLAMA_BASE_URL
        self.model = settings.REASONING_MODEL
        self.timeout = settings.REASONING_TIMEOUT
        self._client: Optional[httpx.AsyncClient] = None
        logger.info(f"ReasoningService initialized with model: {self.model}")
    
    @property
    def client(self) -> httpx.AsyncClient:
        """Lazy initialization of async HTTP client"""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout)
            )
        return self._client
    
    async def close(self):
        """Close the HTTP client"""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
    
    async def check_model_available(self) -> bool:
        """Check if the reasoning model is available"""
        try:
            response = await self.client.get("/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name", "").lower() for m in models]
                
                target = self.model.lower()
                target_base = target.split(":")[0]
                
                logger.debug(f"Looking for reasoning model '{target}' in: {model_names}")
                
                for name in model_names:
                    name_base = name.split(":")[0]
                    if (target == name or 
                        target in name or 
                        target_base == name_base or
                        target_base in name_base or
                        name_base in target_base):
                        logger.info(f"Found matching reasoning model: {name}")
                        return True
                
                logger.warning(f"Reasoning model {self.model} not found. Available: {model_names}")
                return False
            return False
        except Exception as e:
            logger.error(f"Error checking reasoning model: {e}")
            return False
    
    async def _generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = None,
        max_tokens: int = None
    ) -> str:
        """
        Generate text using the reasoning model.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system instruction
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text
        """
        if temperature is None:
            temperature = settings.REASONING_TEMPERATURE
        if max_tokens is None:
            max_tokens = settings.REASONING_MAX_TOKENS
        
        logger.info(f"[REASON] Calling reasoning model: {self.model}")
        logger.info(f"[REASON] Temperature: {temperature}, Max tokens: {max_tokens}")
        logger.debug(f"[REASON] Prompt length: {len(prompt)} chars")
        if system_prompt:
            logger.debug(f"[REASON] System prompt length: {len(system_prompt)} chars")
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        try:
            logger.info(f"[REASON] Sending request to Ollama...")
            start_time = datetime.utcnow()
            
            response = await self.client.post(
                "/api/generate",
                json=payload
            )
            response.raise_for_status()
            
            elapsed = (datetime.utcnow() - start_time).total_seconds()
            result = response.json()
            response_text = result.get("response", "")
            
            logger.info(f"[REASON] Response received in {elapsed:.2f}s")
            logger.info(f"[REASON] Response length: {len(response_text)} chars")
            
            # Log token usage if available
            if "eval_count" in result:
                logger.info(f"[REASON] Tokens generated: {result.get('eval_count')}")
            if "prompt_eval_count" in result:
                logger.info(f"[REASON] Prompt tokens: {result.get('prompt_eval_count')}")
            
            return response_text
            
        except httpx.TimeoutException:
            logger.error(f"[REASON] Timeout after {self.timeout}s - query may be too complex")
            raise
        except httpx.HTTPStatusError as e:
            logger.error(f"[REASON] HTTP error {e.response.status_code}: {e.response.text[:200]}")
            raise
        except Exception as e:
            logger.error(f"[REASON] Error: {type(e).__name__}: {e}")
            raise
        except Exception as e:
            logger.error(f"Reasoning model error: {e}")
            raise
    
    async def answer_question(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        include_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Answer a question using retrieved context chunks.
        
        This is the main RAG function that synthesizes an answer
        from relevant document chunks.
        
        Args:
            query: User's question
            context_chunks: Retrieved relevant chunks with metadata
            include_sources: Whether to cite sources in the answer
            
        Returns:
            Dict with answer, confidence, and sources
        """
        logger.info(f"[REASON] Starting answer generation")
        logger.info(f"[REASON] Query: '{query[:80]}...'")
        logger.info(f"[REASON] Context chunks: {len(context_chunks)}")
        
        start_time = datetime.utcnow()
        
        # Build context from chunks
        context_parts = []
        total_context_len = 0
        for i, chunk in enumerate(context_chunks):
            source_info = f"[Source {i+1}: {chunk.get('filename', 'unknown')}, Page {chunk.get('page_number', '?')}]"
            chunk_text = chunk.get('text', '')
            context_parts.append(f"{source_info}\n{chunk_text}")
            total_context_len += len(chunk_text)
            logger.debug(f"[REASON]   Chunk {i+1}: {len(chunk_text)} chars, score={chunk.get('score', 0):.3f}")
        
        context = "\n\n---\n\n".join(context_parts)
        logger.info(f"[REASON] Total context length: {total_context_len} chars")
        
        system_prompt = """You are a helpful assistant that answers questions based on provided document context.

Rules:
1. Only use information from the provided context
2. If the answer is not in the context, say "I cannot find this information in the provided documents"
3. Be precise and cite which source you got the information from
4. For numerical data, quote the exact values from the source
5. If multiple sources provide different information, note the discrepancy"""

        prompt = f"""Context from documents:
{context}

---

Question: {query}

Please provide a clear, accurate answer based only on the context above. Cite the source numbers when referencing specific information."""

        # Generate answer
        logger.info(f"[REASON] Generating answer...")
        answer = await self._generate(prompt, system_prompt)
        
        # Calculate confidence based on retrieval scores
        scores = [chunk.get("score", 0) for chunk in context_chunks if "score" in chunk]
        avg_score = sum(scores) / len(scores) if scores else 0.5
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        logger.info(f"[REASON] Answer generation COMPLETE")
        logger.info(f"[REASON]   Answer length: {len(answer)} chars")
        logger.info(f"[REASON]   Confidence: {avg_score:.3f}")
        logger.info(f"[REASON]   Processing time: {processing_time:.2f}s")
        
        result = {
            "answer": answer.strip(),
            "confidence": avg_score,
            "model_used": self.model,
            "processing_time": processing_time,
            "num_sources": len(context_chunks)
        }
        
        if include_sources:
            result["sources"] = [
                {
                    "text": chunk.get("text", "")[:500] + "..." if len(chunk.get("text", "")) > 500 else chunk.get("text", ""),
                    "page_number": chunk.get("page_number"),
                    "document_id": chunk.get("document_id"),
                    "filename": chunk.get("filename"),
                    "score": chunk.get("score", 0),
                    "content_type": chunk.get("content_type", "text")
                }
                for chunk in context_chunks
            ]
        
        return result
    
    async def summarize_document(
        self,
        document_text: str,
        max_length: int = 500
    ) -> str:
        """
        Generate a summary of a document.
        
        Args:
            document_text: Full document text
            max_length: Approximate maximum summary length
            
        Returns:
            Document summary
        """
        system_prompt = """You are a document summarization assistant. 
Create concise, informative summaries that capture the key points."""

        prompt = f"""Please summarize the following document in approximately {max_length} characters:

{document_text}

Summary:"""

        return await self._generate(prompt, system_prompt)
    
    async def extract_key_facts(
        self,
        document_text: str
    ) -> List[str]:
        """
        Extract key facts from a document.
        
        Args:
            document_text: Document text
            
        Returns:
            List of key facts
        """
        system_prompt = "You are an information extraction assistant."
        
        prompt = f"""Extract the key facts from this document. 
Return as a JSON array of strings, each containing one fact.

Document:
{document_text}

Key facts (JSON array):"""

        response = await self._generate(prompt, system_prompt)
        
        try:
            # Try to parse as JSON
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]
            
            facts = json.loads(response.strip())
            if isinstance(facts, list):
                return facts
        except json.JSONDecodeError:
            pass
        
        # Fallback: split by newlines
        return [line.strip("- ").strip() for line in response.split("\n") if line.strip()]
    
    async def compare_documents(
        self,
        doc1_text: str,
        doc2_text: str
    ) -> Dict[str, Any]:
        """
        Compare two documents and identify similarities/differences.
        
        Args:
            doc1_text: First document text
            doc2_text: Second document text
            
        Returns:
            Comparison analysis
        """
        system_prompt = "You are a document comparison assistant."
        
        prompt = f"""Compare these two documents and identify:
1. Key similarities
2. Key differences
3. Information present in one but not the other

Document 1:
{doc1_text[:3000]}

Document 2:
{doc2_text[:3000]}

Provide your comparison as JSON:
{{
    "similarities": ["..."],
    "differences": ["..."],
    "only_in_doc1": ["..."],
    "only_in_doc2": ["..."],
    "summary": "..."
}}"""

        response = await self._generate(prompt, system_prompt)
        
        try:
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            return json.loads(response.strip())
        except json.JSONDecodeError:
            return {
                "summary": response,
                "parse_error": True
            }
    
    async def answer_with_schema(
        self,
        query: str,
        context: str,
        output_schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Answer a question with structured output matching a schema.
        
        Args:
            query: User's question
            context: Document context
            output_schema: JSON schema for the output
            
        Returns:
            Structured answer matching the schema
        """
        schema_str = json.dumps(output_schema, indent=2)
        
        system_prompt = f"""You are a structured data extraction assistant.
You must respond with valid JSON matching this schema:
{schema_str}"""

        prompt = f"""Context:
{context}

Question: {query}

Respond with JSON matching the specified schema:"""

        response = await self._generate(prompt, system_prompt, temperature=0.1)
        
        try:
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]
            return json.loads(response.strip())
        except json.JSONDecodeError:
            logger.warning("Failed to parse structured response")
            return {"raw_response": response, "parse_error": True}
    
    async def classify_query_intent(
        self,
        query: str
    ) -> Dict[str, Any]:
        """
        Classify the intent of a user query.
        
        Useful for routing queries to appropriate handlers.
        
        Args:
            query: User's query
            
        Returns:
            Intent classification
        """
        system_prompt = "You are a query classification assistant."
        
        prompt = f"""Classify this user query:
"{query}"

Determine:
1. Intent: (question, extraction, comparison, summary, search, other)
2. Entity types being asked about: (dates, amounts, names, etc.)
3. Specificity: (specific/general)
4. Expected answer type: (factual, analytical, list, yes/no)

Response as JSON:
{{
    "intent": "...",
    "entities": ["..."],
    "specificity": "...",
    "answer_type": "...",
    "confidence": 0.9
}}"""

        response = await self._generate(prompt, system_prompt, temperature=0.1)
        
        try:
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            return json.loads(response.strip())
        except json.JSONDecodeError:
            return {
                "intent": "question",
                "entities": [],
                "specificity": "general",
                "answer_type": "factual",
                "confidence": 0.5
            }
