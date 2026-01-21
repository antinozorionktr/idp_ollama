"""
Tier 1: Vision-Language Model Service (The "Eyes")
Uses Qwen2.5-VL for document understanding and structured extraction

This replaces traditional OCR + Layout Parser with a single VLM that:
- Understands document layout natively
- Extracts tables as structured data
- Converts complex forms to JSON
- Handles multi-page documents
"""

import base64
import json
import logging
import httpx
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import fitz  # PyMuPDF
from PIL import Image
import io
import asyncio
from datetime import datetime

from config import settings

logger = logging.getLogger(__name__)


class VisionService:
    """
    Vision-Language Model service for document understanding.
    Uses Ollama to run Qwen2.5-VL locally.
    """
    
    def __init__(self):
        self.base_url = settings.OLLAMA_BASE_URL
        self.model = settings.VISION_MODEL
        self.timeout = settings.VISION_TIMEOUT
        self._client: Optional[httpx.AsyncClient] = None
        logger.info(f"VisionService initialized with model: {self.model}")
    
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
        """Check if the vision model is available in Ollama"""
        try:
            response = await self.client.get("/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name", "").lower() for m in models]
                
                # Normalize our model name for comparison
                target = self.model.lower()
                target_base = target.split(":")[0]
                
                logger.debug(f"Looking for model '{target}' in available models: {model_names}")
                
                # Check for exact match, partial match, or base name match
                for name in model_names:
                    name_base = name.split(":")[0]
                    if (target == name or 
                        target in name or 
                        target_base == name_base or
                        target_base in name_base or
                        name_base in target_base):
                        logger.info(f"Found matching model: {name} for {self.model}")
                        return True
                
                logger.warning(f"Model {self.model} not found. Available: {model_names}")
                return False
            logger.warning(f"Ollama API returned status {response.status_code}")
            return False
        except Exception as e:
            logger.error(f"Error checking model availability: {e}")
            return False
    
    def _encode_image(self, image_bytes: bytes) -> str:
        """Encode image bytes to base64"""
        return base64.b64encode(image_bytes).decode("utf-8")
    
    def _pdf_to_images(
        self, 
        pdf_content: bytes, 
        dpi: int = None
    ) -> List[bytes]:
        """
        Convert PDF pages to images for vision model processing.
        
        Args:
            pdf_content: PDF file content as bytes
            dpi: Resolution for rendering (default from settings)
            
        Returns:
            List of PNG image bytes, one per page
        """
        if dpi is None:
            dpi = settings.PDF_DPI
        
        images = []
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        
        try:
            # Calculate zoom factor from DPI (72 is default PDF DPI)
            zoom = dpi / 72
            matrix = fitz.Matrix(zoom, zoom)
            
            for page_num in range(min(len(doc), settings.MAX_PAGES_PER_DOC)):
                page = doc[page_num]
                pix = page.get_pixmap(matrix=matrix)
                img_bytes = pix.tobytes("png")
                images.append(img_bytes)
                
            logger.info(f"Converted PDF to {len(images)} images at {dpi} DPI")
            
        finally:
            doc.close()
        
        return images
    
    def _build_extraction_prompt(
        self,
        schema: Optional[Dict[str, Any]] = None,
        custom_prompt: Optional[str] = None
    ) -> str:
        """Build the prompt for document extraction"""
        
        if custom_prompt:
            return custom_prompt
        
        base_prompt = """You are an expert document analyzer. Examine this document image carefully and extract all information.

Your task:
1. Identify the document type (invoice, form, letter, report, etc.)
2. Extract ALL text content, preserving structure
3. Identify and extract any tables as structured data
4. Note any figures, logos, or stamps
5. Preserve numerical values exactly as shown

Output your response as a JSON object with the following structure:
{
    "document_type": "type of document",
    "content": {
        "text": "all extracted text in reading order",
        "sections": [
            {"heading": "section title", "content": "section content"}
        ]
    },
    "tables": [
        {
            "title": "table title if any",
            "headers": ["col1", "col2"],
            "rows": [["val1", "val2"]]
        }
    ],
    "key_value_pairs": {
        "field_name": "value"
    },
    "figures": ["description of any figures/images"],
    "metadata": {
        "language": "detected language",
        "confidence": 0.95
    }
}"""
        
        if schema:
            schema_str = json.dumps(schema, indent=2)
            base_prompt += f"""

IMPORTANT: You must also extract data matching this specific schema:
{schema_str}

Add the schema-matched data under a "structured_data" key in your response."""
        
        base_prompt += "\n\nRespond ONLY with valid JSON, no additional text."
        
        return base_prompt
    
    async def _call_vision_model(
        self,
        images: List[str],  # base64 encoded
        prompt: str,
        temperature: float = None
    ) -> str:
        """
        Call the Ollama vision model with images.
        
        Args:
            images: List of base64-encoded images
            prompt: The extraction prompt
            temperature: Model temperature (default from settings)
            
        Returns:
            Model response text
        """
        if temperature is None:
            temperature = settings.VISION_TEMPERATURE
        
        logger.info(f"[VISION] Calling Ollama vision model: {self.model}")
        logger.info(f"[VISION] Number of images: {len(images)}")
        logger.info(f"[VISION] Temperature: {temperature}")
        logger.info(f"[VISION] Max tokens: {settings.VISION_MAX_TOKENS}")
        logger.debug(f"[VISION] Prompt length: {len(prompt)} chars")
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "images": images,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": settings.VISION_MAX_TOKENS
            }
        }
        
        try:
            logger.info(f"[VISION] Sending request to {self.base_url}/api/generate...")
            start_time = datetime.utcnow()
            
            response = await self.client.post(
                "/api/generate",
                json=payload
            )
            response.raise_for_status()
            
            elapsed = (datetime.utcnow() - start_time).total_seconds()
            result = response.json()
            response_text = result.get("response", "")
            
            logger.info(f"[VISION] Response received in {elapsed:.2f}s")
            logger.info(f"[VISION] Response length: {len(response_text)} chars")
            logger.debug(f"[VISION] Response preview: {response_text[:200]}...")
            
            # Log token usage if available
            if "eval_count" in result:
                logger.info(f"[VISION] Tokens generated: {result.get('eval_count')}")
            if "eval_duration" in result:
                eval_ms = result.get('eval_duration', 0) / 1e6
                logger.info(f"[VISION] Eval duration: {eval_ms:.2f}ms")
            
            return response_text
            
        except httpx.TimeoutException:
            logger.error(f"[VISION] Timeout after {self.timeout}s - model may be loading or document too complex")
            raise
        except httpx.HTTPStatusError as e:
            logger.error(f"[VISION] HTTP error {e.response.status_code}: {e.response.text[:200]}")
            raise
        except Exception as e:
            logger.error(f"[VISION] Error: {type(e).__name__}: {e}")
            raise
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """
        Parse JSON from model response, handling common formatting issues.
        """
        # Remove markdown code blocks if present
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0]
        elif "```" in response:
            response = response.split("```")[1].split("```")[0]
        
        response = response.strip()
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # Try to find JSON object in the response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass
            
            # Return raw text as fallback
            logger.warning("Could not parse JSON response, returning raw text")
            return {
                "document_type": "unknown",
                "content": {"text": response},
                "tables": [],
                "key_value_pairs": {},
                "figures": [],
                "metadata": {"parse_error": True}
            }
    
    async def extract_from_image(
        self,
        image_content: bytes,
        schema: Optional[Dict[str, Any]] = None,
        custom_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract structured data from a single image.
        
        Args:
            image_content: Image file bytes
            schema: Optional JSON schema for extraction
            custom_prompt: Optional custom extraction prompt
            
        Returns:
            Extracted document data as dictionary
        """
        start_time = datetime.utcnow()
        
        # Encode image
        encoded_image = self._encode_image(image_content)
        
        # Build prompt
        prompt = self._build_extraction_prompt(schema, custom_prompt)
        
        # Call vision model
        response = await self._call_vision_model(
            images=[encoded_image],
            prompt=prompt
        )
        
        # Parse response
        result = self._parse_json_response(response)
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        result["_processing_time"] = processing_time
        result["_page_number"] = 1
        
        logger.info(f"Extracted data from image in {processing_time:.2f}s")
        
        return result
    
    async def extract_from_pdf(
        self,
        pdf_content: bytes,
        schema: Optional[Dict[str, Any]] = None,
        custom_prompt: Optional[str] = None,
        batch_size: int = None
    ) -> Dict[str, Any]:
        """
        Extract structured data from a PDF document.
        
        Processes each page through the vision model and combines results.
        
        Args:
            pdf_content: PDF file bytes
            schema: Optional JSON schema for extraction
            custom_prompt: Optional custom extraction prompt
            batch_size: Pages to process in parallel
            
        Returns:
            Combined extraction results from all pages
        """
        start_time = datetime.utcnow()
        
        if batch_size is None:
            batch_size = settings.BATCH_SIZE
        
        # Convert PDF to images
        page_images = self._pdf_to_images(pdf_content)
        num_pages = len(page_images)
        
        logger.info(f"Processing {num_pages} page PDF")
        
        # Build prompt (same for all pages, but could be page-specific)
        prompt = self._build_extraction_prompt(schema, custom_prompt)
        
        # Process pages in batches
        all_page_results = []
        
        for i in range(0, num_pages, batch_size):
            batch_images = page_images[i:i + batch_size]
            
            # Process batch concurrently
            tasks = []
            for j, img_bytes in enumerate(batch_images):
                page_num = i + j + 1
                encoded = self._encode_image(img_bytes)
                
                async def process_page(img: str, pn: int):
                    result = await self._call_vision_model(
                        images=[img],
                        prompt=prompt
                    )
                    parsed = self._parse_json_response(result)
                    parsed["_page_number"] = pn
                    return parsed
                
                tasks.append(process_page(encoded, page_num))
            
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Page processing error: {result}")
                    all_page_results.append({
                        "document_type": "error",
                        "content": {"text": ""},
                        "error": str(result)
                    })
                else:
                    all_page_results.append(result)
        
        # Combine results from all pages
        combined = self._combine_page_results(all_page_results, schema)
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        combined["_processing_time"] = processing_time
        combined["_num_pages"] = num_pages
        
        logger.info(f"Extracted data from {num_pages} pages in {processing_time:.2f}s")
        
        return combined
    
    def _combine_page_results(
        self,
        page_results: List[Dict[str, Any]],
        schema: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Combine extraction results from multiple pages into a single document.
        """
        combined = {
            "document_type": None,
            "content": {
                "text": "",
                "sections": []
            },
            "tables": [],
            "key_value_pairs": {},
            "figures": [],
            "pages": [],
            "metadata": {
                "num_pages": len(page_results),
                "confidence": 0.0
            }
        }
        
        confidences = []
        
        for page_result in page_results:
            page_num = page_result.get("_page_number", 0)
            
            # Determine document type from first valid page
            if not combined["document_type"] and page_result.get("document_type"):
                combined["document_type"] = page_result["document_type"]
            
            # Combine text content
            page_text = page_result.get("content", {}).get("text", "")
            if page_text:
                combined["content"]["text"] += f"\n\n--- Page {page_num} ---\n\n{page_text}"
            
            # Combine sections
            sections = page_result.get("content", {}).get("sections", [])
            for section in sections:
                section["page"] = page_num
                combined["content"]["sections"].append(section)
            
            # Combine tables
            tables = page_result.get("tables", [])
            for table in tables:
                table["page"] = page_num
                combined["tables"].append(table)
            
            # Merge key-value pairs
            kvs = page_result.get("key_value_pairs", {})
            combined["key_value_pairs"].update(kvs)
            
            # Combine figures
            figures = page_result.get("figures", [])
            for fig in figures:
                combined["figures"].append({"page": page_num, "description": fig})
            
            # Track confidence
            conf = page_result.get("metadata", {}).get("confidence", 1.0)
            if isinstance(conf, (int, float)):
                confidences.append(conf)
            
            # Store page-level data
            combined["pages"].append({
                "page_number": page_num,
                "data": page_result
            })
            
            # Handle structured_data from schema extraction
            if "structured_data" in page_result:
                if "structured_data" not in combined:
                    combined["structured_data"] = page_result["structured_data"]
                else:
                    # Merge structured data (simple strategy - could be smarter)
                    self._merge_structured_data(
                        combined["structured_data"],
                        page_result["structured_data"]
                    )
        
        # Calculate average confidence
        if confidences:
            combined["metadata"]["confidence"] = sum(confidences) / len(confidences)
        
        combined["content"]["text"] = combined["content"]["text"].strip()
        
        return combined
    
    def _merge_structured_data(
        self,
        target: Dict[str, Any],
        source: Dict[str, Any]
    ) -> None:
        """Merge structured data from multiple pages"""
        for key, value in source.items():
            if key not in target:
                target[key] = value
            elif isinstance(target[key], list) and isinstance(value, list):
                target[key].extend(value)
            elif isinstance(target[key], dict) and isinstance(value, dict):
                self._merge_structured_data(target[key], value)
            # Otherwise keep the existing value (first page wins)
    
    async def extract_tables_only(
        self,
        image_content: bytes
    ) -> List[Dict[str, Any]]:
        """
        Extract only tables from an image.
        Optimized prompt for table detection and extraction.
        """
        prompt = """Extract all tables from this document image.

For each table, provide:
1. Table title/caption if visible
2. Column headers
3. All row data

Output as JSON:
{
    "tables": [
        {
            "title": "table title or null",
            "headers": ["col1", "col2", "col3"],
            "rows": [
                ["row1col1", "row1col2", "row1col3"],
                ["row2col1", "row2col2", "row2col3"]
            ],
            "notes": "any footnotes or notes"
        }
    ]
}

If no tables are found, return: {"tables": []}
Respond ONLY with valid JSON."""
        
        encoded = self._encode_image(image_content)
        response = await self._call_vision_model(
            images=[encoded],
            prompt=prompt
        )
        
        result = self._parse_json_response(response)
        return result.get("tables", [])
    
    async def classify_document(
        self,
        image_content: bytes
    ) -> Dict[str, Any]:
        """
        Quickly classify a document without full extraction.
        """
        prompt = """Classify this document image. Identify:
1. Document type (invoice, receipt, contract, form, letter, report, ID, certificate, etc.)
2. Language
3. Key entities visible (company names, person names, dates)
4. Is it handwritten or printed?
5. Quality assessment (clear, blurry, partial)

Output as JSON:
{
    "document_type": "type",
    "language": "detected language",
    "entities": ["entity1", "entity2"],
    "is_handwritten": false,
    "quality": "clear",
    "confidence": 0.95
}

Respond ONLY with valid JSON."""
        
        encoded = self._encode_image(image_content)
        response = await self._call_vision_model(
            images=[encoded],
            prompt=prompt,
            temperature=0.1
        )
        
        return self._parse_json_response(response)
