"""
OCR Service v2.0 - Upgraded with High-DPI Vision
- 200+ DPI for better text extraction
- Enhanced table recognition
- Layout-aware processing
"""

from paddleocr import PaddleOCR
import numpy as np
from PIL import Image
import io
import fitz  # PyMuPDF
from typing import List, Dict, Any, Tuple, Optional
import logging
from config import settings

logger = logging.getLogger(__name__)


class OCRService:
    def __init__(self):
        """Initialize PaddleOCR with enhanced settings"""
        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang=settings.PADDLE_OCR_LANG,
            use_gpu=settings.PADDLE_OCR_USE_GPU,
            show_log=False,
            # Enhanced OCR settings
            det_db_thresh=0.3,  # Lower threshold for better detection
            det_db_box_thresh=0.5,
            det_db_unclip_ratio=1.6,
            rec_batch_num=16,
            use_space_char=True,  # Better space detection
        )
        
        # Table structure recognition (optional)
        self.table_engine = None
        if settings.OCR_ENABLE_TABLE_RECOGNITION:
            try:
                from paddleocr import PPStructure
                self.table_engine = PPStructure(
                    show_log=False,
                    use_gpu=settings.PADDLE_OCR_USE_GPU,
                    table=True,
                    ocr=True,
                    layout=settings.OCR_ENABLE_LAYOUT_ANALYSIS
                )
                logger.info("Table structure recognition enabled")
            except Exception as e:
                logger.warning(f"Table recognition not available: {e}")
        
        # DPI setting for rendering
        self.dpi = settings.OCR_DPI  # 200+ DPI
        self.zoom_factor = self.dpi / 72.0  # PDF default is 72 DPI
        
        logger.info(f"OCR Service initialized with {self.dpi} DPI")
    
    def extract_text(self, file_content: bytes, content_type: str) -> Dict[str, Any]:
        """
        Extract text from PDF or image using PaddleOCR with high DPI
        
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
            
            # Convert to PIL Image for preprocessing
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            # Preprocess image for better OCR
            img = self._preprocess_image(img)
            
            # Convert back to bytes
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='PNG', dpi=(self.dpi, self.dpi))
            img_bytes = img_bytes.getvalue()
            
            # Run OCR on high-DPI image
            result = self.ocr.ocr(img_bytes, cls=True)
            page_data = self._parse_ocr_result(result, page_num + 1)
            
            # Extract tables if enabled
            if self.table_engine:
                table_result = self._extract_tables(img_bytes, page_num + 1)
                page_data["tables"] = table_result
                all_tables.extend(table_result)
            
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
        # Open and preprocess image
        img = Image.open(io.BytesIO(img_content))
        
        # Upscale if needed to meet DPI requirements
        img = self._upscale_image(img)
        img = self._preprocess_image(img)
        
        # Convert to bytes
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes = img_bytes.getvalue()
        
        # Run OCR
        result = self.ocr.ocr(img_bytes, cls=True)
        page_data = self._parse_ocr_result(result, 1)
        
        # Extract tables if enabled
        tables = []
        if self.table_engine:
            tables = self._extract_tables(img_bytes, 1)
            page_data["tables"] = tables
        
        return {
            "text": page_data["text"],
            "pages": [page_data],
            "boxes": page_data["boxes"],
            "tables": tables,
            "num_pages": 1,
            "dpi": self.dpi,
            "has_tables": len(tables) > 0
        }
    
    def _preprocess_image(self, img: Image.Image) -> Image.Image:
        """Preprocess image for better OCR accuracy"""
        import cv2
        import numpy as np
        
        # Convert PIL to OpenCV
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding for better text contrast
        # binary = cv2.adaptiveThreshold(
        #     gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        # )
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        
        # Sharpen
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(denoised, -1, kernel)
        
        # Convert back to RGB PIL Image
        img_processed = Image.fromarray(cv2.cvtColor(sharpened, cv2.COLOR_GRAY2RGB))
        
        return img_processed
    
    def _upscale_image(self, img: Image.Image, target_dpi: int = None) -> Image.Image:
        """Upscale image to target DPI if needed"""
        if target_dpi is None:
            target_dpi = self.dpi
        
        # Get current DPI (assume 72 if not specified)
        current_dpi = img.info.get('dpi', (72, 72))[0]
        
        if current_dpi < target_dpi:
            scale_factor = target_dpi / current_dpi
            new_size = (int(img.width * scale_factor), int(img.height * scale_factor))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            logger.info(f"Upscaled image from {current_dpi} to {target_dpi} DPI")
        
        return img
    
    def _extract_tables(self, img_bytes: bytes, page_num: int) -> List[Dict[str, Any]]:
        """Extract tables using PPStructure"""
        tables = []
        
        try:
            result = self.table_engine(img_bytes)
            
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
    
    def _parse_ocr_result(self, result: List, page_num: int) -> Dict[str, Any]:
        """
        Parse PaddleOCR result into structured format
        
        PaddleOCR returns: [[[bbox], (text, confidence)], ...]
        """
        if not result or not result[0]:
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
        
        for line in result[0]:
            bbox = line[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            text_info = line[1]  # (text, confidence)
            
            text = text_info[0]
            confidence = text_info[1]
            
            # Convert bbox to simple format
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
                "polygon": bbox,  # Keep original polygon for precise positioning
                "page": page_num
            }
            
            boxes.append(box_data)
            texts.append(text)
            confidences.append(confidence)
        
        # Sort boxes by vertical position (top to bottom, left to right)
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
