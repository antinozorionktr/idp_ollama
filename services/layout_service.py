"""
Layout Analysis Service using LayoutParser
"""

import numpy as np
from PIL import Image
import io
import fitz
from typing import List, Dict, Any
import logging
from config import settings

logger = logging.getLogger(__name__)

# Try to import LayoutParser, but make it optional
try:
    import layoutparser as lp
    LAYOUTPARSER_AVAILABLE = True
    logger.info("LayoutParser available")
except Exception as e:
    LAYOUTPARSER_AVAILABLE = False
    logger.warning(f"LayoutParser not available: {str(e)}. Using fallback mode.")


class LayoutService:
    def __init__(self):
        """Initialize LayoutParser model if available"""
        self.model = None
        
        if LAYOUTPARSER_AVAILABLE:
            try:
                self.model = lp.Detectron2LayoutModel(
                    config_path=settings.LAYOUT_MODEL,
                    extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 
                                 settings.LAYOUT_CONFIDENCE_THRESHOLD],
                    label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}
                )
                logger.info("LayoutParser model initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize LayoutParser: {str(e)}. Using fallback mode.")
                self.model = None
        else:
            logger.info("LayoutParser not available, using fallback mode")
    
    def analyze_layout(
        self, 
        file_content: bytes, 
        content_type: str, 
        ocr_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze document layout to identify tables, text blocks, figures, etc.
        Falls back to basic layout if LayoutParser is not available.
        """
        try:
            if self.model is not None:
                # Use LayoutParser if available
                if content_type == 'application/pdf':
                    return self._analyze_pdf_layout(file_content, ocr_result)
                else:
                    return self._analyze_image_layout(file_content, ocr_result)
            else:
                # Use fallback mode
                logger.info("Using fallback layout analysis (no table detection)")
                return self._create_basic_layout(ocr_result)
        except Exception as e:
            logger.error(f"Layout analysis failed: {str(e)}")
            return self._create_basic_layout(ocr_result)
    
    def _analyze_pdf_layout(
        self, 
        pdf_content: bytes, 
        ocr_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze layout for PDF"""
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        all_layouts = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Convert to image
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            # Detect layout
            if self.model:
                layout = self.model.detect(np.array(img))
                page_layout = self._process_layout(layout, page_num + 1, ocr_result)
            else:
                page_layout = self._create_basic_page_layout(page_num + 1, ocr_result)
            
            all_layouts.append(page_layout)
        
        doc.close()
        
        # Aggregate results
        has_tables = any(page.get("has_tables", False) for page in all_layouts)
        has_figures = any(page.get("has_figures", False) for page in all_layouts)
        
        return {
            "pages": all_layouts,
            "has_tables": has_tables,
            "has_figures": has_figures,
            "num_pages": len(all_layouts)
        }
    
    def _analyze_image_layout(
        self, 
        img_content: bytes, 
        ocr_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze layout for image"""
        img = Image.open(io.BytesIO(img_content))
        
        if self.model:
            layout = self.model.detect(np.array(img))
            page_layout = self._process_layout(layout, 1, ocr_result)
        else:
            page_layout = self._create_basic_page_layout(1, ocr_result)
        
        return {
            "pages": [page_layout],
            "has_tables": page_layout.get("has_tables", False),
            "has_figures": page_layout.get("has_figures", False),
            "num_pages": 1
        }
    
    def _process_layout(
        self, 
        layout: lp.Layout, 
        page_num: int, 
        ocr_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process detected layout elements"""
        blocks = []
        tables = []
        figures = []
        text_regions = []
        
        # Get OCR boxes for this page
        page_ocr_boxes = [
            box for box in ocr_result.get("boxes", []) 
            if box.get("page") == page_num
        ]
        
        for block in layout:
            block_dict = {
                "type": block.type,
                "bbox": {
                    "x1": block.block.x_1,
                    "y1": block.block.y_1,
                    "x2": block.block.x_2,
                    "y2": block.block.y_2
                },
                "confidence": float(block.score),
                "page": page_num
            }
            
            # Assign OCR text to blocks
            block_dict["text"] = self._get_text_in_block(
                block_dict["bbox"], 
                page_ocr_boxes
            )
            
            blocks.append(block_dict)
            
            if block.type == "Table":
                tables.append(block_dict)
            elif block.type == "Figure":
                figures.append(block_dict)
            elif block.type in ["Text", "Title", "List"]:
                text_regions.append(block_dict)
        
        # Sort blocks by reading order (top to bottom, left to right)
        blocks.sort(key=lambda b: (b["bbox"]["y1"], b["bbox"]["x1"]))
        
        return {
            "page": page_num,
            "blocks": blocks,
            "tables": tables,
            "figures": figures,
            "text_regions": text_regions,
            "has_tables": len(tables) > 0,
            "has_figures": len(figures) > 0
        }
    
    def _get_text_in_block(
        self, 
        block_bbox: Dict[str, float], 
        ocr_boxes: List[Dict[str, Any]]
    ) -> str:
        """Get OCR text that falls within a layout block"""
        texts = []
        
        for ocr_box in ocr_boxes:
            ocr_bbox = ocr_box["bbox"]
            
            # Check if OCR box overlaps with block
            if self._boxes_overlap(block_bbox, ocr_bbox):
                texts.append(ocr_box["text"])
        
        return " ".join(texts)
    
    def _boxes_overlap(self, box1: Dict, box2: Dict) -> bool:
        """Check if two bounding boxes overlap"""
        return not (
            box1["x2"] < box2["x1"] or
            box1["x1"] > box2["x2"] or
            box1["y2"] < box2["y1"] or
            box1["y1"] > box2["y2"]
        )
    
    def _create_basic_layout(self, ocr_result: Dict[str, Any]) -> Dict[str, Any]:
        """Create basic layout when LayoutParser is not available"""
        pages = []
        
        for page_data in ocr_result.get("pages", []):
            page_num = page_data.get("page", 1)
            page_layout = self._create_basic_page_layout(page_num, ocr_result)
            pages.append(page_layout)
        
        return {
            "pages": pages,
            "has_tables": False,
            "has_figures": False,
            "num_pages": len(pages)
        }
    
    def _create_basic_page_layout(
        self, 
        page_num: int, 
        ocr_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create basic page layout from OCR boxes"""
        page_boxes = [
            box for box in ocr_result.get("boxes", [])
            if box.get("page") == page_num
        ]
        
        # Group OCR boxes into text blocks
        text_block = {
            "type": "Text",
            "text": " ".join([box["text"] for box in page_boxes]),
            "bbox": self._compute_combined_bbox(page_boxes),
            "confidence": 1.0,
            "page": page_num
        }
        
        return {
            "page": page_num,
            "blocks": [text_block],
            "tables": [],
            "figures": [],
            "text_regions": [text_block],
            "has_tables": False,
            "has_figures": False
        }
    
    def _compute_combined_bbox(self, boxes: List[Dict]) -> Dict[str, float]:
        """Compute combined bounding box for multiple boxes"""
        if not boxes:
            return {"x1": 0, "y1": 0, "x2": 0, "y2": 0}
        
        x1 = min(box["bbox"]["x1"] for box in boxes)
        y1 = min(box["bbox"]["y1"] for box in boxes)
        x2 = max(box["bbox"]["x2"] for box in boxes)
        y2 = max(box["bbox"]["y2"] for box in boxes)
        
        return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
