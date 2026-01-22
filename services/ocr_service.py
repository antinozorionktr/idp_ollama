"""
OCR Service v2.0 - Upgraded with High-DPI Vision
Compatible with PaddleOCR v3.x API
- 200+ DPI for better text extraction
- CPU compatibility fixes for Docker/VM environments
- Fallback to Tesseract if PaddleOCR fails
"""

from PIL import Image
import numpy as np
import io
import fitz  # PyMuPDF
from typing import List, Dict, Any, Optional
import logging
import os

logger = logging.getLogger(__name__)

# ============================================================
# CRITICAL: Set these BEFORE importing PaddleOCR
# Fixes "Illegal instruction" (SIGILL) error in Docker/VM
# ============================================================
os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_ir_optim"] = "0"  # Disable IR optimization that causes SIGILL
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["PADDLE_MKL_NUM_THREADS"] = "4"


class OCRService:
    def __init__(self):
        """Initialize PaddleOCR with CPU-safe settings"""
        self.ocr = None
        self.table_engine = None
        self.use_tesseract = False
        self.dpi = 200  # Default DPI
        self.zoom_factor = self.dpi / 72.0
        
        # Try to load settings
        try:
            from config import settings
            self.dpi = getattr(settings, 'OCR_DPI', 200)
            self.zoom_factor = self.dpi / 72.0
            lang = getattr(settings, 'PADDLE_OCR_LANG', 'en')
            enable_table = getattr(settings, 'OCR_ENABLE_TABLE_RECOGNITION', False)
        except ImportError:
            lang = "en"
            enable_table = False
        
        # Initialize PaddleOCR with CPU-safe settings
        self._init_ocr(lang)
        
        # Initialize table engine if enabled and PaddleOCR worked
        if enable_table and self.ocr is not None:
            self._init_table_engine()
        
        logger.info(f"OCR Service initialized with {self.dpi} DPI")
    
    def _init_ocr(self, lang: str = "en"):
        """Initialize PaddleOCR with multiple fallback options"""
        
        # Attempt 1: PaddleOCR with minimal CPU-safe settings
        try:
            from paddleocr import PaddleOCR
            
            self.ocr = PaddleOCR(
                use_angle_cls=False,  # Disable to reduce complexity
                lang=lang,
                use_gpu=False,
                show_log=False,
                enable_mkldnn=False,  # Disable MKL-DNN
                cpu_threads=4,
            )
            
            # Test with a small image to verify it works
            test_img = np.zeros((100, 100, 3), dtype=np.uint8)
            test_img.fill(255)
            _ = self.ocr.ocr(test_img, cls=False)
            
            logger.info(f"PaddleOCR initialized successfully with lang={lang}")
            return
            
        except Exception as e:
            logger.warning(f"PaddleOCR failed: {e}")
            self.ocr = None
        
        # Attempt 2: Tesseract fallback
        self._init_tesseract_fallback()
    
    def _init_tesseract_fallback(self):
        """Initialize Tesseract as fallback OCR"""
        try:
            import pytesseract
            # Test tesseract is available
            pytesseract.get_tesseract_version()
            self.use_tesseract = True
            logger.info("Using Tesseract as fallback OCR engine")
        except Exception as e:
            logger.error(f"Tesseract also not available: {e}")
            logger.error("No OCR engine available! Install tesseract-ocr or fix PaddlePaddle")
            self.use_tesseract = False
    
    def _init_table_engine(self):
        """Initialize table structure recognition"""
        try:
            from paddleocr import PPStructure
            self.table_engine = PPStructure(
                table=True, 
                ocr=True,
                show_log=False,
                use_gpu=False,
                enable_mkldnn=False,
            )
            logger.info("Table structure recognition enabled")
        except Exception as e:
            logger.warning(f"Table recognition not available: {e}")
            self.table_engine = None
    
    def extract_text(self, file_content: bytes, content_type: str) -> Dict[str, Any]:
        """
        Extract text from PDF or image
        
        Args:
            file_content: File content as bytes
            content_type: MIME type of the file
            
        Returns:
            Dictionary containing extracted text and metadata
        """
        try:
            if content_type == 'application/pdf':
                return self._process_pdf(file_content)
            else:
                return self._process_image(file_content)
        except Exception as e:
            logger.error(f"OCR extraction failed: {str(e)}")
            raise
    
    def _process_pdf(self, pdf_content: bytes) -> Dict[str, Any]:
        """Process PDF file page by page with high DPI rendering"""
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        all_pages = []
        all_tables = []
        
        # Create transformation matrix for high DPI
        matrix = fitz.Matrix(self.zoom_factor, self.zoom_factor)
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Render page at high DPI (200+)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            
            # Convert to PIL Image
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            # Preprocess image for better OCR
            img = self._preprocess_image(img)
            
            # Run OCR
            page_data = self._run_ocr(img, page_num + 1)
            
            # Extract tables if enabled
            if self.table_engine:
                try:
                    table_result = self._extract_tables(np.array(img), page_num + 1)
                    page_data["tables"] = table_result
                    all_tables.extend(table_result)
                except Exception as e:
                    logger.warning(f"Table extraction failed for page {page_num + 1}: {e}")
                    page_data["tables"] = []
            
            all_pages.append(page_data)
        
        doc.close()
        
        # Combine all pages
        full_text = "\n\n".join([page["text"] for page in all_pages])
        all_boxes = [box for page in all_pages for box in page["boxes"]]
        
        return {
            "text": full_text,
            "pages": all_pages,
            "boxes": all_boxes,
            "tables": all_tables,
            "num_pages": len(all_pages),
            "dpi": self.dpi,
            "has_tables": len(all_tables) > 0
        }
    
    def _process_image(self, img_content: bytes) -> Dict[str, Any]:
        """Process image file with preprocessing"""
        # Open image
        img = Image.open(io.BytesIO(img_content))
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Upscale if needed
        img = self._upscale_image(img)
        img = self._preprocess_image(img)
        
        # Run OCR
        page_data = self._run_ocr(img, 1)
        
        # Extract tables if enabled
        tables = []
        if self.table_engine:
            try:
                tables = self._extract_tables(np.array(img), 1)
                page_data["tables"] = tables
            except Exception as e:
                logger.warning(f"Table extraction failed: {e}")
        
        return {
            "text": page_data["text"],
            "pages": [page_data],
            "boxes": page_data["boxes"],
            "tables": tables,
            "num_pages": 1,
            "dpi": self.dpi,
            "has_tables": len(tables) > 0
        }
    
    def _run_ocr(self, img: Image.Image, page_num: int) -> Dict[str, Any]:
        """Run OCR on image using available engine"""
        img_array = np.array(img)
        
        # Try PaddleOCR first
        if self.ocr is not None:
            try:
                result = self.ocr.ocr(img_array, cls=False)
                return self._parse_paddle_result(result, page_num)
            except Exception as e:
                logger.warning(f"PaddleOCR execution failed: {e}")
        
        # Fallback to Tesseract
        if self.use_tesseract:
            return self._run_tesseract(img, page_num)
        
        # Last resort: return empty result
        logger.error("No OCR engine available")
        return {
            "page": page_num,
            "text": "",
            "boxes": [],
            "word_count": 0,
            "avg_confidence": 0.0
        }
    
    def _run_tesseract(self, img: Image.Image, page_num: int) -> Dict[str, Any]:
        """Run Tesseract OCR as fallback"""
        try:
            import pytesseract
            
            # Get detailed data
            data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
            
            boxes = []
            texts = []
            confidences = []
            
            for i in range(len(data['text'])):
                text = data['text'][i].strip()
                conf = int(data['conf'][i])
                
                if text and conf > 0:
                    box_data = {
                        "text": text,
                        "confidence": conf / 100.0,
                        "bbox": {
                            "x1": data['left'][i],
                            "y1": data['top'][i],
                            "x2": data['left'][i] + data['width'][i],
                            "y2": data['top'][i] + data['height'][i]
                        },
                        "page": page_num
                    }
                    boxes.append(box_data)
                    texts.append(text)
                    confidences.append(conf / 100.0)
            
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            return {
                "page": page_num,
                "text": " ".join(texts),
                "boxes": boxes,
                "word_count": len(texts),
                "avg_confidence": avg_confidence
            }
            
        except Exception as e:
            logger.error(f"Tesseract OCR failed: {e}")
            return {
                "page": page_num,
                "text": "",
                "boxes": [],
                "word_count": 0,
                "avg_confidence": 0.0
            }
    
    def _preprocess_image(self, img: Image.Image) -> Image.Image:
        """Preprocess image for better OCR accuracy"""
        try:
            import cv2
            
            # Convert PIL to OpenCV
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            
            # Convert to grayscale
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            
            # Denoise
            denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
            
            # Sharpen
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(denoised, -1, kernel)
            
            # Convert back to RGB PIL Image
            img_processed = Image.fromarray(cv2.cvtColor(sharpened, cv2.COLOR_GRAY2RGB))
            
            return img_processed
            
        except ImportError:
            logger.warning("OpenCV not available, skipping preprocessing")
            return img
        except Exception as e:
            logger.warning(f"Preprocessing failed: {e}, using original image")
            return img
    
    def _upscale_image(self, img: Image.Image, target_dpi: int = None) -> Image.Image:
        """Upscale image to target DPI if needed"""
        if target_dpi is None:
            target_dpi = self.dpi
        
        # Get current DPI (assume 72 if not specified)
        try:
            current_dpi = img.info.get('dpi', (72, 72))[0]
        except:
            current_dpi = 72
        
        if current_dpi < target_dpi:
            scale_factor = target_dpi / current_dpi
            new_size = (int(img.width * scale_factor), int(img.height * scale_factor))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            logger.debug(f"Upscaled image from {current_dpi} to {target_dpi} DPI")
        
        return img
    
    def _extract_tables(self, img_array: np.ndarray, page_num: int) -> List[Dict[str, Any]]:
        """Extract tables using PPStructure"""
        tables = []
        
        if not self.table_engine:
            return tables
        
        try:
            result = self.table_engine(img_array)
            
            for idx, item in enumerate(result):
                if item.get('type') == 'table':
                    table_data = {
                        "table_id": f"table_{page_num}_{idx}",
                        "page": page_num,
                        "bbox": item.get('bbox', []),
                        "html": item.get('res', {}).get('html', ''),
                        "cells": item.get('res', {}).get('cells', []),
                        "confidence": item.get('score', 0.0)
                    }
                    tables.append(table_data)
                    
        except Exception as e:
            logger.warning(f"Table extraction failed: {e}")
        
        return tables
    
    def _parse_paddle_result(self, result: List, page_num: int) -> Dict[str, Any]:
        """Parse PaddleOCR result into structured format"""
        if not result or (isinstance(result, list) and len(result) == 0):
            return {
                "page": page_num,
                "text": "",
                "boxes": [],
                "word_count": 0,
                "avg_confidence": 0.0
            }
        
        # Handle nested list structure
        ocr_lines = result[0] if result and isinstance(result[0], list) else result
        
        if not ocr_lines:
            return {
                "page": page_num,
                "text": "",
                "boxes": [],
                "word_count": 0,
                "avg_confidence": 0.0
            }
        
        boxes = []
        texts = []
        confidences = []
        
        for line in ocr_lines:
            if not line or len(line) < 2:
                continue
                
            bbox = line[0]
            text_info = line[1]
            
            if isinstance(text_info, tuple) and len(text_info) >= 2:
                text = text_info[0]
                confidence = text_info[1]
            elif isinstance(text_info, str):
                text = text_info
                confidence = 1.0
            else:
                continue
            
            try:
                x_coords = [point[0] for point in bbox]
                y_coords = [point[1] for point in bbox]
                
                box_data = {
                    "text": text,
                    "confidence": float(confidence),
                    "bbox": {
                        "x1": min(x_coords),
                        "y1": min(y_coords),
                        "x2": max(x_coords),
                        "y2": max(y_coords)
                    },
                    "polygon": bbox,
                    "page": page_num
                }
                
                boxes.append(box_data)
                texts.append(text)
                confidences.append(confidence)
            except Exception as e:
                logger.warning(f"Failed to parse bbox: {e}")
                continue
        
        # Sort boxes by reading order
        boxes.sort(key=lambda b: (b["bbox"]["y1"], b["bbox"]["x1"]))
        texts = [box["text"] for box in boxes]
        
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return {
            "page": page_num,
            "text": " ".join(texts),
            "boxes": boxes,
            "word_count": len(texts),
            "avg_confidence": avg_confidence
        }
    
    def get_text_with_positions(self, ocr_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get text with spatial positions for layout analysis"""
        return ocr_result.get("boxes", [])