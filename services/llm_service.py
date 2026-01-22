"""
LLM Service v2.0 - Large Local Models
- Ollama integration for DeepSeek-R1 32B, Llama 3.3 70B
- vLLM support for high-throughput inference
- Cloud API fallback (Claude, GPT-4)
"""

import httpx
import json
from typing import Dict, Any, Optional, List, Generator
import logging
from config import settings, LLMProvider

logger = logging.getLogger(__name__)


class LLMService:
    def __init__(self):
        """Initialize LLM clients based on configuration"""
        self.provider = settings.LLM_PROVIDER
        self.ollama_client = None
        self.anthropic_client = None
        self.openai_client = None
        
        # Initialize based on provider
        if self.provider == LLMProvider.OLLAMA:
            self._init_ollama()
        elif self.provider == LLMProvider.VLLM:
            self._init_vllm()
        elif self.provider == LLMProvider.ANTHROPIC:
            self._init_anthropic()
        elif self.provider == LLMProvider.OPENAI:
            self._init_openai()
        
        # Always initialize cloud APIs as fallback
        self._init_anthropic()
        self._init_openai()
        
        logger.info(f"LLM Service initialized with provider: {self.provider}")
    
    def _init_ollama(self):
        """Initialize Ollama client"""
        try:
            # Test Ollama connection
            response = httpx.get(
                f"{settings.OLLAMA_BASE_URL}/api/tags",
                timeout=10
            )
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m["name"] for m in models]
                logger.info(f"Ollama connected. Available models: {model_names}")
                
                # Check if target model is available
                if settings.OLLAMA_MODEL not in model_names:
                    logger.warning(f"Model {settings.OLLAMA_MODEL} not found. Available: {model_names}")
            else:
                logger.warning("Ollama connection failed")
        except Exception as e:
            logger.warning(f"Ollama not available: {e}")
    
    def _init_vllm(self):
        """Initialize vLLM client"""
        try:
            response = httpx.get(
                f"{settings.VLLM_BASE_URL}/v1/models",
                timeout=10
            )
            if response.status_code == 200:
                logger.info("vLLM server connected")
            else:
                logger.warning("vLLM connection failed")
        except Exception as e:
            logger.warning(f"vLLM not available: {e}")
    
    def _init_anthropic(self):
        """Initialize Anthropic client"""
        if settings.ANTHROPIC_API_KEY:
            try:
                import anthropic
                self.anthropic_client = anthropic.Anthropic(
                    api_key=settings.ANTHROPIC_API_KEY
                )
                logger.info("Anthropic client initialized")
            except Exception as e:
                logger.warning(f"Anthropic init failed: {e}")
    
    def _init_openai(self):
        """Initialize OpenAI client"""
        if settings.OPENAI_API_KEY:
            try:
                import openai
                openai.api_key = settings.OPENAI_API_KEY
                self.openai_client = openai
                logger.info("OpenAI client initialized")
            except Exception as e:
                logger.warning(f"OpenAI init failed: {e}")
    
    def extract_structured_data(
        self,
        context: str,
        schema: Optional[str] = None,
        use_local: bool = True
    ) -> Dict[str, Any]:
        """
        Extract structured data from context using LLM
        
        Args:
            context: Document text to extract from
            schema: JSON schema definition (as string or dict)
            use_local: Use local LLM (Ollama/vLLM) vs cloud API
            
        Returns:
            Extracted structured data as dictionary
        """
        try:
            # Parse schema if string
            if schema and isinstance(schema, str):
                try:
                    schema = json.loads(schema)
                except json.JSONDecodeError:
                    logger.warning("Invalid JSON schema, proceeding without schema")
                    schema = None
            
            # Build prompt
            system_prompt = self._build_extraction_prompt(schema)
            user_message = f"""Document Content:
{context}

Please extract all relevant information according to the schema and return as JSON only."""

            # Choose provider
            if use_local and self.provider in [LLMProvider.OLLAMA, LLMProvider.VLLM]:
                response_text = self._call_local_llm(system_prompt, user_message)
            elif self.anthropic_client:
                response_text = self._call_anthropic(system_prompt, user_message)
            elif self.openai_client:
                response_text = self._call_openai(system_prompt, user_message)
            else:
                raise ValueError("No LLM provider available")
            
            # Parse JSON response
            return self._parse_json_response(response_text)
            
        except Exception as e:
            logger.error(f"Error in structured extraction: {str(e)}")
            raise
    
    def generate_answer(
        self,
        query: str,
        context: str,
        use_local: bool = True
    ) -> str:
        """
        Generate an answer to a query based on context
        """
        try:
            system_prompt = """You are a helpful assistant that answers questions based on the provided context.
            
Rules:
1. Only use information from the provided context
2. If the answer is not in the context, say so
3. Be concise and accurate
4. Cite specific parts of the context when relevant"""

            user_message = f"""Context:
{context}

Question: {query}

Please provide a clear and accurate answer based only on the context provided."""

            # Choose provider
            if use_local and self.provider in [LLMProvider.OLLAMA, LLMProvider.VLLM]:
                return self._call_local_llm(system_prompt, user_message)
            elif self.anthropic_client:
                return self._call_anthropic(system_prompt, user_message)
            elif self.openai_client:
                return self._call_openai(system_prompt, user_message)
            else:
                raise ValueError("No LLM provider available")
                
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            raise
    
    def _call_local_llm(self, system_prompt: str, user_message: str) -> str:
        """Call Ollama or vLLM"""
        if self.provider == LLMProvider.OLLAMA:
            return self._call_ollama(system_prompt, user_message)
        else:
            return self._call_vllm(system_prompt, user_message)
    
    def _call_ollama(self, system_prompt: str, user_message: str) -> str:
        """Call Ollama API"""
        try:
            response = httpx.post(
                f"{settings.OLLAMA_BASE_URL}/api/chat",
                json={
                    "model": settings.OLLAMA_MODEL,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    "stream": False,
                    "options": {
                        "temperature": settings.TEMPERATURE,
                        "num_predict": settings.MAX_TOKENS
                    }
                },
                timeout=settings.OLLAMA_TIMEOUT
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["message"]["content"]
            else:
                raise Exception(f"Ollama error: {response.text}")
                
        except Exception as e:
            logger.error(f"Ollama call failed: {e}")
            # Fallback to cloud
            if self.anthropic_client:
                logger.info("Falling back to Anthropic")
                return self._call_anthropic(system_prompt, user_message)
            raise
    
    def _call_vllm(self, system_prompt: str, user_message: str) -> str:
        """Call vLLM OpenAI-compatible API"""
        try:
            response = httpx.post(
                f"{settings.VLLM_BASE_URL}/v1/chat/completions",
                json={
                    "model": settings.VLLM_MODEL,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    "temperature": settings.TEMPERATURE,
                    "max_tokens": settings.MAX_TOKENS
                },
                timeout=settings.OLLAMA_TIMEOUT
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                raise Exception(f"vLLM error: {response.text}")
                
        except Exception as e:
            logger.error(f"vLLM call failed: {e}")
            # Fallback to cloud
            if self.anthropic_client:
                return self._call_anthropic(system_prompt, user_message)
            raise
    
    def _call_anthropic(self, system_prompt: str, user_message: str) -> str:
        """Call Anthropic Claude API"""
        try:
            response = self.anthropic_client.messages.create(
                model=settings.CLAUDE_MODEL,
                max_tokens=settings.MAX_TOKENS,
                temperature=settings.TEMPERATURE,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_message}
                ]
            )
            
            return response.content[0].text
            
        except Exception as e:
            logger.error(f"Anthropic call failed: {e}")
            raise
    
    def _call_openai(self, system_prompt: str, user_message: str) -> str:
        """Call OpenAI GPT-4 API"""
        try:
            response = self.openai_client.ChatCompletion.create(
                model=settings.GPT_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=settings.TEMPERATURE,
                max_tokens=settings.MAX_TOKENS
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"OpenAI call failed: {e}")
            raise
    
    def _build_extraction_prompt(self, schema: Optional[Dict[str, Any]] = None) -> str:
        """Build system prompt for structured extraction"""
        base_prompt = """You are an expert at extracting structured information from documents.

Your task is to extract all relevant information from the provided text and return it in a structured JSON format.

Rules:
1. Extract all information accurately
2. Preserve exact values, numbers, dates, and names
3. Return ONLY valid JSON, no additional text or markdown
4. Use null for missing information
5. Follow the provided schema exactly if given
6. Think step by step before extracting"""

        if schema:
            schema_str = json.dumps(schema, indent=2)
            base_prompt += f"\n\nExtraction Schema:\n{schema_str}"
            base_prompt += "\n\nYou MUST follow this exact schema structure."
        
        return base_prompt
    
    def _parse_json_response(self, response_text: str) -> Dict[str, Any]:
        """Parse JSON from LLM response"""
        # Remove markdown code blocks
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]
        
        # Clean up
        response_text = response_text.strip()
        
        # Try to parse
        try:
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {response_text[:200]}...")
            # Try to find JSON object in text
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except:
                    pass
            raise ValueError(f"Could not parse JSON from response: {str(e)}")
    
    def stream_generate(
        self,
        prompt: str,
        system_prompt: str = None
    ) -> Generator[str, None, None]:
        """Stream generation for real-time output"""
        if self.provider == LLMProvider.OLLAMA:
            try:
                with httpx.stream(
                    "POST",
                    f"{settings.OLLAMA_BASE_URL}/api/chat",
                    json={
                        "model": settings.OLLAMA_MODEL,
                        "messages": [
                            {"role": "system", "content": system_prompt or "You are a helpful assistant."},
                            {"role": "user", "content": prompt}
                        ],
                        "stream": True
                    },
                    timeout=settings.OLLAMA_TIMEOUT
                ) as response:
                    for line in response.iter_lines():
                        if line:
                            data = json.loads(line)
                            if "message" in data:
                                yield data["message"].get("content", "")
            except Exception as e:
                logger.error(f"Stream generation failed: {e}")
                yield f"Error: {str(e)}"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current LLM configuration"""
        info = {
            "provider": self.provider.value,
            "temperature": settings.TEMPERATURE,
            "max_tokens": settings.MAX_TOKENS
        }
        
        if self.provider == LLMProvider.OLLAMA:
            info["model"] = settings.OLLAMA_MODEL
            info["base_url"] = settings.OLLAMA_BASE_URL
        elif self.provider == LLMProvider.VLLM:
            info["model"] = settings.VLLM_MODEL
            info["base_url"] = settings.VLLM_BASE_URL
        elif self.provider == LLMProvider.ANTHROPIC:
            info["model"] = settings.CLAUDE_MODEL
        elif self.provider == LLMProvider.OPENAI:
            info["model"] = settings.GPT_MODEL
        
        return info
    
    def list_available_models(self) -> List[str]:
        """List available models from Ollama"""
        if self.provider == LLMProvider.OLLAMA:
            try:
                response = httpx.get(
                    f"{settings.OLLAMA_BASE_URL}/api/tags",
                    timeout=10
                )
                if response.status_code == 200:
                    models = response.json().get("models", [])
                    return [m["name"] for m in models]
            except:
                pass
        return []
