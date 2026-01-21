"""
Chunking Service
Intelligent document segmentation for RAG

Works with Vision Service output to create optimal chunks for:
- Vector embedding
- Context retrieval
- Answer generation
"""

import logging
import re
from typing import List, Dict, Any, Optional
import uuid

from config import settings
from models.schemas import ContentType

logger = logging.getLogger(__name__)


class ChunkingService:
    """
    Document chunking service that creates semantic chunks
    from vision-extracted content.
    """
    
    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None
    ):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        logger.info(
            f"ChunkingService initialized: size={self.chunk_size}, "
            f"overlap={self.chunk_overlap}"
        )
    
    def create_chunks(
        self,
        extracted_data: Dict[str, Any],
        document_id: str,
        filename: str
    ) -> List[Dict[str, Any]]:
        """
        Create chunks from vision-extracted document data.
        
        Handles:
        - Regular text with sentence-aware splitting
        - Tables as single chunks (preserved whole)
        - Sections as semantic units
        
        Args:
            extracted_data: Output from VisionService
            document_id: Unique document identifier
            filename: Original filename
            
        Returns:
            List of chunk dictionaries
        """
        all_chunks = []
        chunk_counter = 0
        
        # Process pages if available
        pages = extracted_data.get("pages", [])
        
        if pages:
            for page_data in pages:
                page_num = page_data.get("page_number", 1)
                page_content = page_data.get("data", {})
                
                # Create chunks from page content
                page_chunks = self._process_page(
                    page_content,
                    page_num,
                    document_id,
                    filename,
                    chunk_counter
                )
                
                all_chunks.extend(page_chunks)
                chunk_counter += len(page_chunks)
        else:
            # Single page/image document
            page_chunks = self._process_page(
                extracted_data,
                1,
                document_id,
                filename,
                chunk_counter
            )
            all_chunks.extend(page_chunks)
        
        logger.info(f"Created {len(all_chunks)} chunks from document {document_id}")
        return all_chunks
    
    def _process_page(
        self,
        page_content: Dict[str, Any],
        page_num: int,
        document_id: str,
        filename: str,
        chunk_start: int
    ) -> List[Dict[str, Any]]:
        """Process a single page into chunks"""
        chunks = []
        chunk_id = chunk_start
        
        # 1. Process tables (keep as single chunks)
        tables = page_content.get("tables", [])
        for i, table in enumerate(tables):
            table_text = self._table_to_text(table)
            if table_text.strip():
                chunks.append({
                    "chunk_id": str(chunk_id),
                    "text": table_text,
                    "content_type": ContentType.TABLE.value,
                    "page_number": page_num,
                    "document_id": document_id,
                    "filename": filename,
                    "metadata": {
                        "table_index": i,
                        "table_title": table.get("title"),
                        "is_table": True
                    }
                })
                chunk_id += 1
        
        # 2. Process sections
        sections = page_content.get("content", {}).get("sections", [])
        for section in sections:
            section_text = f"{section.get('heading', '')}\n{section.get('content', '')}"
            section_chunks = self._chunk_text(
                section_text.strip(),
                ContentType.TEXT,
                page_num,
                document_id,
                filename,
                chunk_id,
                {"section_heading": section.get("heading")}
            )
            chunks.extend(section_chunks)
            chunk_id += len(section_chunks)
        
        # 3. Process main text (if no sections)
        if not sections:
            main_text = page_content.get("content", {}).get("text", "")
            if not main_text:
                main_text = page_content.get("text", "")
            
            if main_text.strip():
                text_chunks = self._chunk_text(
                    main_text.strip(),
                    ContentType.TEXT,
                    page_num,
                    document_id,
                    filename,
                    chunk_id
                )
                chunks.extend(text_chunks)
                chunk_id += len(text_chunks)
        
        # 4. Process key-value pairs as a chunk
        kvs = page_content.get("key_value_pairs", {})
        if kvs:
            kv_text = "\n".join([f"{k}: {v}" for k, v in kvs.items()])
            chunks.append({
                "chunk_id": str(chunk_id),
                "text": kv_text,
                "content_type": ContentType.TEXT.value,
                "page_number": page_num,
                "document_id": document_id,
                "filename": filename,
                "metadata": {
                    "is_key_value_section": True,
                    "keys": list(kvs.keys())
                }
            })
        
        return chunks
    
    def _table_to_text(self, table: Dict[str, Any]) -> str:
        """Convert table data to readable text format"""
        lines = []
        
        if table.get("title"):
            lines.append(f"Table: {table['title']}")
            lines.append("")
        
        headers = table.get("headers", [])
        rows = table.get("rows", [])
        
        if headers:
            lines.append(" | ".join(str(h) for h in headers))
            lines.append("-" * 40)
        
        for row in rows:
            lines.append(" | ".join(str(cell) for cell in row))
        
        if table.get("notes"):
            lines.append("")
            lines.append(f"Notes: {table['notes']}")
        
        return "\n".join(lines)
    
    def _chunk_text(
        self,
        text: str,
        content_type: ContentType,
        page_num: int,
        document_id: str,
        filename: str,
        chunk_start: int,
        extra_metadata: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Split text into chunks with sentence awareness.
        """
        if not text or len(text) <= self.chunk_size:
            if not text:
                return []
            return [{
                "chunk_id": str(chunk_start),
                "text": text,
                "content_type": content_type.value,
                "page_number": page_num,
                "document_id": document_id,
                "filename": filename,
                "metadata": extra_metadata or {}
            }]
        
        # Split into sentences
        sentences = self._split_sentences(text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        chunk_id = chunk_start
        
        for sentence in sentences:
            sentence_len = len(sentence)
            
            # If single sentence exceeds chunk size, split by words
            if sentence_len > self.chunk_size:
                if current_chunk:
                    chunks.append(self._create_chunk(
                        " ".join(current_chunk),
                        content_type,
                        page_num,
                        document_id,
                        filename,
                        chunk_id,
                        extra_metadata
                    ))
                    chunk_id += 1
                    current_chunk = []
                    current_length = 0
                
                # Split long sentence
                word_chunks = self._split_long_sentence(
                    sentence,
                    content_type,
                    page_num,
                    document_id,
                    filename,
                    chunk_id,
                    extra_metadata
                )
                chunks.extend(word_chunks)
                chunk_id += len(word_chunks)
                continue
            
            # Check if adding this sentence exceeds limit
            if current_length + sentence_len > self.chunk_size:
                # Save current chunk
                if current_chunk:
                    chunks.append(self._create_chunk(
                        " ".join(current_chunk),
                        content_type,
                        page_num,
                        document_id,
                        filename,
                        chunk_id,
                        extra_metadata
                    ))
                    chunk_id += 1
                
                # Start new chunk with overlap
                overlap_sentences = self._get_overlap_sentences(
                    current_chunk
                )
                current_chunk = overlap_sentences + [sentence]
                current_length = sum(len(s) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_length += sentence_len
        
        # Add remaining chunk
        if current_chunk:
            chunks.append(self._create_chunk(
                " ".join(current_chunk),
                content_type,
                page_num,
                document_id,
                filename,
                chunk_id,
                extra_metadata
            ))
        
        return chunks
    
    def _create_chunk(
        self,
        text: str,
        content_type: ContentType,
        page_num: int,
        document_id: str,
        filename: str,
        chunk_id: int,
        extra_metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Create a chunk dictionary"""
        return {
            "chunk_id": str(chunk_id),
            "text": text.strip(),
            "content_type": content_type.value,
            "page_number": page_num,
            "document_id": document_id,
            "filename": filename,
            "metadata": extra_metadata or {}
        }
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences with abbreviation handling"""
        # Protect common abbreviations
        abbreviations = [
            ("Dr.", "Dr<DOT>"), ("Mr.", "Mr<DOT>"), ("Mrs.", "Mrs<DOT>"),
            ("Ms.", "Ms<DOT>"), ("Jr.", "Jr<DOT>"), ("Sr.", "Sr<DOT>"),
            ("Inc.", "Inc<DOT>"), ("Ltd.", "Ltd<DOT>"), ("Corp.", "Corp<DOT>"),
            ("etc.", "etc<DOT>"), ("vs.", "vs<DOT>"), ("i.e.", "i<DOT>e<DOT>"),
            ("e.g.", "e<DOT>g<DOT>"), ("No.", "No<DOT>"), ("Vol.", "Vol<DOT>"),
            ("Fig.", "Fig<DOT>"), ("pp.", "pp<DOT>")
        ]
        
        protected = text
        for abbr, replacement in abbreviations:
            protected = protected.replace(abbr, replacement)
        
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', protected)
        
        # Restore abbreviations
        restored = []
        for sentence in sentences:
            for abbr, replacement in abbreviations:
                sentence = sentence.replace(replacement, abbr)
            if sentence.strip():
                restored.append(sentence.strip())
        
        return restored
    
    def _split_long_sentence(
        self,
        sentence: str,
        content_type: ContentType,
        page_num: int,
        document_id: str,
        filename: str,
        chunk_start: int,
        extra_metadata: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """Split a long sentence by words"""
        words = sentence.split()
        chunks = []
        current_words = []
        current_length = 0
        chunk_id = chunk_start
        
        for word in words:
            word_len = len(word) + 1  # +1 for space
            
            if current_length + word_len > self.chunk_size:
                if current_words:
                    chunks.append(self._create_chunk(
                        " ".join(current_words),
                        content_type,
                        page_num,
                        document_id,
                        filename,
                        chunk_id,
                        {**(extra_metadata or {}), "split_sentence": True}
                    ))
                    chunk_id += 1
                current_words = [word]
                current_length = len(word)
            else:
                current_words.append(word)
                current_length += word_len
        
        if current_words:
            chunks.append(self._create_chunk(
                " ".join(current_words),
                content_type,
                page_num,
                document_id,
                filename,
                chunk_id,
                {**(extra_metadata or {}), "split_sentence": True}
            ))
        
        return chunks
    
    def _get_overlap_sentences(
        self,
        sentences: List[str]
    ) -> List[str]:
        """Get sentences for overlap from the end of a chunk"""
        if not sentences:
            return []
        
        overlap_text = ""
        overlap_sentences = []
        
        for sentence in reversed(sentences):
            if len(overlap_text) + len(sentence) <= self.chunk_overlap:
                overlap_sentences.insert(0, sentence)
                overlap_text = " ".join(overlap_sentences)
            else:
                break
        
        return overlap_sentences
    
    def get_chunk_statistics(
        self,
        chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Get statistics about created chunks"""
        if not chunks:
            return {
                "total_chunks": 0,
                "avg_length": 0,
                "by_type": {}
            }
        
        by_type = {}
        total_length = 0
        
        for chunk in chunks:
            ctype = chunk.get("content_type", "text")
            by_type[ctype] = by_type.get(ctype, 0) + 1
            total_length += len(chunk.get("text", ""))
        
        return {
            "total_chunks": len(chunks),
            "avg_length": total_length / len(chunks),
            "min_length": min(len(c.get("text", "")) for c in chunks),
            "max_length": max(len(c.get("text", "")) for c in chunks),
            "by_type": by_type,
            "by_page": self._count_by_page(chunks)
        }
    
    def _count_by_page(
        self,
        chunks: List[Dict[str, Any]]
    ) -> Dict[int, int]:
        """Count chunks by page number"""
        by_page = {}
        for chunk in chunks:
            page = chunk.get("page_number", 1)
            by_page[page] = by_page.get(page, 0) + 1
        return by_page
