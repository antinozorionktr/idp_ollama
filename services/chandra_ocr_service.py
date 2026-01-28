"""
Chandra OCR Service - State-of-the-art document OCR
- Handles complex tables, forms, handwriting with full layout
- Outputs Markdown, HTML, or JSON with bounding boxes
- Supports local (HuggingFace) and remote (vLLM) inference
- Best-in-class accuracy on olmocr benchmark (83.1%)
"""

import io
import logging
import os
from typing import List, Dict, Any, Optional, Union
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


class ChandraOCRService:
    """
    OCR Service using Chandra - highly accurate OCR for complex documents.
    
    Features:
    - Tables with merged cells (colspan/rowspan)
    - Handwriting recognition
    - Mathematical equations (LaTeX output)
    - Forms with checkboxes
    - Multi-column layouts
    - 40+ language support
    """
    
    def __init__(
        self, 
        method: str = "hf",  # "hf" for HuggingFace local, "vllm" for vLLM server
        model_checkpoint: str = "datalab-to/chandra",
        vllm_api_base: str = "http://localhost:8000/v1",
        max_output_tokens: int = 8192,
        dpi: int = 300,
        include_images: bool = True,
        include_headers_footers: bool = False
    ):
        """
        Initialize Chandra OCR service.
        
        Args:
            method: Inference method - "hf" (HuggingFace local) or "vllm" (vLLM server)
            model_checkpoint: Model name on HuggingFace
            vllm_api_base: vLLM server URL if using vllm method
            max_output_tokens: Maximum tokens per page
            dpi: DPI for PDF rendering
            include_images: Extract and return images from documents
            include_headers_footers: Include page headers/footers in output
        """
        self.method = method
        self.model_checkpoint = model_checkpoint
        self.vllm_api_base = vllm_api_base
        self.max_output_tokens = max_output_tokens
        self.dpi = dpi
        self.include_images = include_images
        self.include_headers_footers = include_headers_footers
        
        self.manager = None
        self.model = None
        self._initialized = False
        
        # Initialize the OCR engine
        self._init_ocr()
    
    def _init_ocr(self):
        """Initialize the Chandra OCR engine"""
        try:
            if self.method == "hf":
                self._init_huggingface()
            elif self.method == "vllm":
                self._init_vllm()
            else:
                raise ValueError(f"Unknown method: {self.method}. Use 'hf' or 'vllm'")
            
            self._initialized = True
            logger.info(f"Chandra OCR initialized with method: {self.method}")
            
        except ImportError as e:
            logger.error(f"Failed to import Chandra: {e}")
            logger.error("Install with: pip install chandra-ocr")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Chandra OCR: {e}")
            raise
    
    def _init_huggingface(self):
        """Initialize with HuggingFace Transformers (local inference)"""
        try:
            from chandra.model import InferenceManager
            
            self.manager = InferenceManager(method="hf")
            logger.info("Chandra HuggingFace model loaded successfully")
            
        except ImportError:
            # Try alternative import for direct model loading
            logger.info("Attempting direct model loading...")
            from transformers import AutoModel, AutoProcessor
            
            self.model = AutoModel.from_pretrained(self.model_checkpoint)
            self.model.processor = AutoProcessor.from_pretrained(self.model_checkpoint)
            
            # Move to GPU if available
            import torch
            if torch.cuda.is_available():
                self.model = self.model.cuda()
                logger.info("Chandra model loaded on CUDA")
            else:
                logger.info("Chandra model loaded on CPU")
    
    def _init_vllm(self):
        """Initialize with vLLM server (remote inference)"""
        from chandra.model import InferenceManager
        
        # Set environment variables for vLLM
        os.environ["VLLM_API_BASE"] = self.vllm_api_base
        
        self.manager = InferenceManager(method="vllm")
        logger.info(f"Chandra vLLM client connected to {self.vllm_api_base}")
    
    def extract_text(self, file_content: bytes, content_type: str) -> Dict[str, Any]:
        """
        Extract text from PDF or image using Chandra OCR.
        
        Args:
            file_content: File content as bytes
            content_type: MIME type of the file
            
        Returns:
            Dictionary containing:
            - text: Full extracted text
            - markdown: Markdown formatted output
            - html: HTML formatted output with layout
            - pages: Per-page results with bounding boxes
            - tables: Extracted table data
            - num_pages: Number of pages processed
        """
        if not self._initialized:
            raise RuntimeError("Chandra OCR not initialized")
        
        try:
            if content_type == 'application/pdf':
                return self._process_pdf(file_content)
            else:
                return self._process_image(file_content)
        except Exception as e:
            logger.error(f"Chandra OCR extraction failed: {e}")
            raise
    
    def _process_pdf(self, pdf_content: bytes) -> Dict[str, Any]:
        """Process PDF file using Chandra"""
        try:
            from chandra.input import load_pdf_images
            
            # Save PDF temporarily to process
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
                f.write(pdf_content)
                temp_path = f.name
            
            try:
                # Load PDF as images
                images = load_pdf_images(temp_path, dpi=self.dpi)
                
                # Process all pages
                results = self._process_images(images)
                
                return self._combine_results(results)
                
            finally:
                # Clean up temp file
                os.unlink(temp_path)
                
        except ImportError:
            # Fallback: Use PyMuPDF for PDF to image conversion
            return self._process_pdf_fallback(pdf_content)
    
    def _process_pdf_fallback(self, pdf_content: bytes) -> Dict[str, Any]:
        """Fallback PDF processing using PyMuPDF"""
        import fitz  # PyMuPDF
        
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        images = []
        
        # Create transformation matrix for target DPI
        zoom_factor = self.dpi / 72.0
        matrix = fitz.Matrix(zoom_factor, zoom_factor)
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
        
        doc.close()
        
        # Process all pages
        results = self._process_images(images)
        return self._combine_results(results)
    
    def _process_image(self, img_content: bytes) -> Dict[str, Any]:
        """Process single image using Chandra"""
        # Open image
        img = Image.open(io.BytesIO(img_content))
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Process single image
        results = self._process_images([img])
        return self._combine_results(results)
    
    def _process_images(self, images: List[Image.Image]) -> List[Dict[str, Any]]:
        """Process a list of images through Chandra"""
        results = []
        
        if self.manager is not None:
            # Use InferenceManager
            try:
                from chandra.model.schema import BatchInputItem
                
                batch = [
                    BatchInputItem(
                        image=img,
                        prompt_type="ocr_layout"
                    )
                    for img in images
                ]
                
                ocr_results = self.manager.generate(batch)
                
                for page_num, result in enumerate(ocr_results, 1):
                    page_data = self._parse_chandra_result(result, page_num)
                    results.append(page_data)
                    
            except Exception as e:
                logger.error(f"Chandra batch processing failed: {e}")
                # Fallback to individual processing
                for page_num, img in enumerate(images, 1):
                    page_data = self._process_single_image(img, page_num)
                    results.append(page_data)
        else:
            # Direct model inference
            for page_num, img in enumerate(images, 1):
                page_data = self._process_single_image(img, page_num)
                results.append(page_data)
        
        return results
    
    def _process_single_image(self, img: Image.Image, page_num: int) -> Dict[str, Any]:
        """Process a single image using direct model inference"""
        try:
            from chandra.model.hf import generate_hf
            from chandra.model.schema import BatchInputItem
            from chandra.output import parse_markdown
            
            batch = [BatchInputItem(image=img, prompt_type="ocr_layout")]
            result = generate_hf(batch, self.model)[0]
            
            return self._parse_chandra_result(result, page_num)
            
        except Exception as e:
            logger.error(f"Single image processing failed: {e}")
            return {
                "page": page_num,
                "text": "",
                "markdown": "",
                "html": "",
                "boxes": [],
                "tables": [],
                "word_count": 0,
                "avg_confidence": 0.0,
                "error": str(e)
            }
    
    def _parse_chandra_result(self, result, page_num: int) -> Dict[str, Any]:
        """Parse Chandra result into structured format"""
        # Extract markdown text
        markdown = getattr(result, 'markdown', '') or ''
        
        # Extract HTML if available
        html = getattr(result, 'html', '') or ''
        
        # Extract plain text (strip markdown formatting)
        text = self._markdown_to_text(markdown)
        
        # Extract bounding boxes if available
        boxes = []
        layout_blocks = getattr(result, 'layout', []) or getattr(result, 'blocks', []) or []
        
        for block in layout_blocks:
            if isinstance(block, dict):
                box_data = {
                    "text": block.get('text', ''),
                    "confidence": block.get('confidence', 1.0),
                    "bbox": block.get('bbox', {}),
                    "type": block.get('type', 'text'),
                    "page": page_num
                }
                boxes.append(box_data)
        
        # Extract tables
        tables = []
        table_blocks = [b for b in layout_blocks if isinstance(b, dict) and b.get('type') == 'table']
        for idx, table in enumerate(table_blocks):
            tables.append({
                "table_id": f"table_{page_num}_{idx}",
                "page": page_num,
                "bbox": table.get('bbox', {}),
                "html": table.get('html', ''),
                "markdown": table.get('markdown', ''),
                "cells": table.get('cells', [])
            })
        
        # Calculate metrics
        words = text.split()
        word_count = len(words)
        
        return {
            "page": page_num,
            "text": text,
            "markdown": markdown,
            "html": html,
            "boxes": boxes,
            "tables": tables,
            "word_count": word_count,
            "avg_confidence": 1.0,  # Chandra doesn't expose per-word confidence
            "raw_result": getattr(result, 'raw', None)
        }
    
    def _markdown_to_text(self, markdown: str) -> str:
        """Convert markdown to plain text"""
        import re
        
        if not markdown:
            return ""
        
        text = markdown
        
        # Remove images
        text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
        
        # Remove links but keep text
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
        
        # Remove headers
        text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
        
        # Remove bold/italic
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
        text = re.sub(r'__([^_]+)__', r'\1', text)
        text = re.sub(r'_([^_]+)_', r'\1', text)
        
        # Remove code blocks
        text = re.sub(r'```[\s\S]*?```', '', text)
        text = re.sub(r'`([^`]+)`', r'\1', text)
        
        # Clean up whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = text.strip()
        
        return text
    
    def _combine_results(self, page_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine per-page results into final output"""
        if not page_results:
            return {
                "text": "",
                "markdown": "",
                "html": "",
                "pages": [],
                "boxes": [],
                "tables": [],
                "num_pages": 0,
                "dpi": self.dpi,
                "has_tables": False,
                "ocr_engine": "chandra"
            }
        
        # Combine text
        full_text = "\n\n".join([p["text"] for p in page_results])
        full_markdown = "\n\n---\n\n".join([p["markdown"] for p in page_results])
        
        # Combine HTML
        html_pages = [p.get("html", "") for p in page_results]
        full_html = "\n".join([f'<div class="page" data-page="{i+1}">{h}</div>' 
                               for i, h in enumerate(html_pages)])
        
        # Collect all boxes and tables
        all_boxes = [box for page in page_results for box in page.get("boxes", [])]
        all_tables = [table for page in page_results for table in page.get("tables", [])]
        
        return {
            "text": full_text,
            "markdown": full_markdown,
            "html": full_html,
            "pages": page_results,
            "boxes": all_boxes,
            "tables": all_tables,
            "num_pages": len(page_results),
            "dpi": self.dpi,
            "has_tables": len(all_tables) > 0,
            "ocr_engine": "chandra"
        }
    
    def get_text_with_positions(self, ocr_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get text with spatial positions for layout analysis"""
        return ocr_result.get("boxes", [])
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the OCR model"""
        return {
            "engine": "chandra",
            "model": self.model_checkpoint,
            "method": self.method,
            "dpi": self.dpi,
            "max_output_tokens": self.max_output_tokens,
            "vllm_api_base": self.vllm_api_base if self.method == "vllm" else None,
            "initialized": self._initialized
        }


# Factory function to create OCR service based on config
def create_ocr_service(config=None):
    """
    Factory function to create the appropriate OCR service.
    
    Args:
        config: Configuration object with OCR settings
        
    Returns:
        OCR service instance (ChandraOCRService or OCRService)
    """
    if config is None:
        try:
            from config import settings
            config = settings
        except ImportError:
            config = None
    
    # Check which OCR engine to use
    ocr_engine = getattr(config, 'OCR_ENGINE', 'chandra').lower() if config else 'chandra'
    
    if ocr_engine == 'chandra':
        try:
            return ChandraOCRService(
                method=getattr(config, 'CHANDRA_METHOD', 'hf'),
                dpi=getattr(config, 'OCR_DPI', 300),
                max_output_tokens=getattr(config, 'CHANDRA_MAX_TOKENS', 8192),
                vllm_api_base=getattr(config, 'CHANDRA_VLLM_URL', 'http://localhost:8000/v1')
            )
        except Exception as e:
            logger.warning(f"Failed to initialize Chandra OCR: {e}, falling back to PaddleOCR")
            from services.ocr_service import OCRService
            return OCRService()
    else:
        # Use original PaddleOCR service
        from services.ocr_service import OCRService
        return OCRService()
