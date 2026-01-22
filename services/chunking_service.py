"""
Chunking Service with Table-Aware Strategy
"""

from typing import List, Dict, Any
import logging
import re

logger = logging.getLogger(__name__)


class ChunkingService:
    def __init__(self):
        """Initialize chunking service"""
        logger.info("Chunking service initialized")
    
    def create_chunks(
        self,
        layout_result: Dict[str, Any],
        chunk_size: int = 512,
        chunk_overlap: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Create chunks with table-aware strategy
        
        Strategy:
        1. Tables are kept as single chunks (not split)
        2. Text blocks are chunked with overlap
        3. Figures are kept separate with captions
        4. Maintain reading order and context
        
        Args:
            layout_result: Layout analysis result
            chunk_size: Maximum characters per chunk
            chunk_overlap: Overlap between chunks
            
        Returns:
            List of chunks with metadata
        """
        all_chunks = []
        chunk_id = 0
        
        for page in layout_result.get("pages", []):
            page_num = page.get("page", 1)
            blocks = page.get("blocks", [])
            
            # Sort blocks by reading order
            blocks.sort(key=lambda b: (b["bbox"]["y1"], b["bbox"]["x1"]))
            
            for block in blocks:
                block_type = block.get("type", "Text")
                text = block.get("text", "").strip()
                
                if not text:
                    continue
                
                if block_type == "Table":
                    # Keep tables as single chunks
                    chunks = self._create_table_chunk(
                        block, page_num, chunk_id
                    )
                    all_chunks.extend(chunks)
                    chunk_id += len(chunks)
                    
                elif block_type == "Figure":
                    # Keep figures with captions
                    chunks = self._create_figure_chunk(
                        block, page_num, chunk_id
                    )
                    all_chunks.extend(chunks)
                    chunk_id += len(chunks)
                    
                else:
                    # Regular text chunking with overlap
                    chunks = self._create_text_chunks(
                        text, 
                        block_type,
                        page_num, 
                        chunk_id,
                        chunk_size, 
                        chunk_overlap
                    )
                    all_chunks.extend(chunks)
                    chunk_id += len(chunks)
        
        logger.info(f"Created {len(all_chunks)} chunks from {layout_result.get('num_pages', 0)} pages")
        return all_chunks
    
    def _create_table_chunk(
        self, 
        table_block: Dict[str, Any], 
        page_num: int,
        chunk_id: int
    ) -> List[Dict[str, Any]]:
        """Create a chunk for a table (no splitting)"""
        return [{
            "chunk_id": chunk_id,
            "text": table_block["text"],
            "type": "table",
            "page": page_num,
            "bbox": table_block["bbox"],
            "metadata": {
                "is_table": True,
                "confidence": table_block.get("confidence", 1.0),
                "block_type": "Table"
            }
        }]
    
    def _create_figure_chunk(
        self, 
        figure_block: Dict[str, Any], 
        page_num: int,
        chunk_id: int
    ) -> List[Dict[str, Any]]:
        """Create a chunk for a figure"""
        return [{
            "chunk_id": chunk_id,
            "text": f"[FIGURE] {figure_block['text']}",
            "type": "figure",
            "page": page_num,
            "bbox": figure_block["bbox"],
            "metadata": {
                "is_figure": True,
                "confidence": figure_block.get("confidence", 1.0),
                "block_type": "Figure"
            }
        }]
    
    def _create_text_chunks(
        self,
        text: str,
        block_type: str,
        page_num: int,
        start_chunk_id: int,
        chunk_size: int,
        chunk_overlap: int
    ) -> List[Dict[str, Any]]:
        """
        Create overlapping text chunks from regular text blocks
        
        Uses sentence-aware chunking to avoid breaking mid-sentence
        """
        # Split into sentences
        sentences = self._split_into_sentences(text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        chunk_id = start_chunk_id
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # If single sentence exceeds chunk_size, split it
            if sentence_length > chunk_size:
                if current_chunk:
                    # Save current chunk
                    chunks.append({
                        "chunk_id": chunk_id,
                        "text": " ".join(current_chunk),
                        "type": "text",
                        "page": page_num,
                        "metadata": {
                            "is_table": False,
                            "is_figure": False,
                            "block_type": block_type,
                            "num_sentences": len(current_chunk)
                        }
                    })
                    chunk_id += 1
                    current_chunk = []
                    current_length = 0
                
                # Split long sentence by words
                words = sentence.split()
                word_chunk = []
                word_length = 0
                
                for word in words:
                    word_length += len(word) + 1
                    if word_length > chunk_size:
                        if word_chunk:
                            chunks.append({
                                "chunk_id": chunk_id,
                                "text": " ".join(word_chunk),
                                "type": "text",
                                "page": page_num,
                                "metadata": {
                                    "is_table": False,
                                    "is_figure": False,
                                    "block_type": block_type,
                                    "split_sentence": True
                                }
                            })
                            chunk_id += 1
                        word_chunk = [word]
                        word_length = len(word)
                    else:
                        word_chunk.append(word)
                
                if word_chunk:
                    current_chunk = word_chunk
                    current_length = word_length
                
            elif current_length + sentence_length > chunk_size:
                # Save current chunk and start new one with overlap
                if current_chunk:
                    chunks.append({
                        "chunk_id": chunk_id,
                        "text": " ".join(current_chunk),
                        "type": "text",
                        "page": page_num,
                        "metadata": {
                            "is_table": False,
                            "is_figure": False,
                            "block_type": block_type,
                            "num_sentences": len(current_chunk)
                        }
                    })
                    chunk_id += 1
                
                # Start new chunk with overlap
                overlap_text = " ".join(current_chunk)
                if len(overlap_text) > chunk_overlap:
                    # Keep last sentences for overlap
                    overlap_sentences = []
                    overlap_length = 0
                    for sent in reversed(current_chunk):
                        overlap_length += len(sent)
                        if overlap_length > chunk_overlap:
                            break
                        overlap_sentences.insert(0, sent)
                    current_chunk = overlap_sentences + [sentence]
                else:
                    current_chunk = current_chunk + [sentence]
                
                current_length = sum(len(s) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        # Add remaining chunk
        if current_chunk:
            chunks.append({
                "chunk_id": chunk_id,
                "text": " ".join(current_chunk),
                "type": "text",
                "page": page_num,
                "metadata": {
                    "is_table": False,
                    "is_figure": False,
                    "block_type": block_type,
                    "num_sentences": len(current_chunk)
                }
            })
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences
        
        Simple sentence splitter that handles common abbreviations
        """
        # Replace abbreviations temporarily
        text = text.replace("Dr.", "Dr<dot>")
        text = text.replace("Mr.", "Mr<dot>")
        text = text.replace("Mrs.", "Mrs<dot>")
        text = text.replace("Ms.", "Ms<dot>")
        text = text.replace("Jr.", "Jr<dot>")
        text = text.replace("Sr.", "Sr<dot>")
        text = text.replace("Inc.", "Inc<dot>")
        text = text.replace("Ltd.", "Ltd<dot>")
        text = text.replace("etc.", "etc<dot>")
        text = text.replace("vs.", "vs<dot>")
        text = text.replace("e.g.", "e<dot>g<dot>")
        text = text.replace("i.e.", "i<dot>e<dot>")
        
        # Split on sentence boundaries
        sentences = re.split(r'[.!?]+\s+', text)
        
        # Restore abbreviations
        sentences = [
            s.replace("<dot>", ".").strip() 
            for s in sentences 
            if s.strip()
        ]
        
        return sentences
    
    def get_chunk_statistics(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about chunks"""
        total_chunks = len(chunks)
        table_chunks = sum(1 for c in chunks if c["metadata"].get("is_table", False))
        figure_chunks = sum(1 for c in chunks if c["metadata"].get("is_figure", False))
        text_chunks = total_chunks - table_chunks - figure_chunks
        
        avg_length = sum(len(c["text"]) for c in chunks) / total_chunks if total_chunks > 0 else 0
        
        return {
            "total_chunks": total_chunks,
            "text_chunks": text_chunks,
            "table_chunks": table_chunks,
            "figure_chunks": figure_chunks,
            "avg_chunk_length": avg_length
        }
