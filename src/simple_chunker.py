# src/simple_chunker.py
import tiktoken
from typing import List, Dict

class SimpleChunker:
    def __init__(self, chunk_size: int = 500, overlap_size: int = 100):
        """
        Initialize the chunker with token-based sizes.
        
        Args:
            chunk_size: Target size in tokens (default 500)
            overlap_size: Overlap between chunks in tokens (default 100)
        """
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        # Use OpenAI's tokenizer (same as text-embedding-3-large uses)
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def chunk_text(self, text: str, metadata: Dict = None) -> List[Dict]:
        """
        Split text into chunks based on token count.
        
        Args:
            text: Text to chunk
            metadata: Base metadata to include with each chunk
            
        Returns:
            List of chunks with metadata
        """
        if not text or not text.strip():
            return []
        
        # Tokenize the entire text
        tokens = self.tokenizer.encode(text)
        
        chunks = []
        chunk_index = 0
        start_idx = 0
        
        while start_idx < len(tokens):
            # Get chunk tokens
            end_idx = min(start_idx + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start_idx:end_idx]
            
            # Decode back to text
            chunk_text = self.tokenizer.decode(chunk_tokens)
            
            # Create chunk with metadata
            chunk_data = {
                "text": chunk_text,
                "chunk_index": chunk_index,
                "start_token": start_idx,
                "end_token": end_idx,
                "token_count": len(chunk_tokens)
            }
            
            # Add any provided metadata
            if metadata:
                chunk_data.update(metadata)
            
            chunks.append(chunk_data)
            
            # Move to next chunk with overlap
            start_idx = end_idx - self.overlap_size
            chunk_index += 1
            
            # Prevent infinite loop if overlap >= chunk_size
            if start_idx <= chunk_index * self.overlap_size:
                start_idx = end_idx
        
        return chunks
    
    def chunk_page_content(self, page_content: str, page_number: int, source_document: str) -> List[Dict]:
        """
        Chunk a page's content, preserving special elements.
        
        Args:
            page_content: The page text content
            page_number: Page number
            source_document: Source document name
            
        Returns:
            List of chunks with metadata
        """
        # First, handle special elements to avoid splitting them
        protected_elements = []
        
        # Find all [TABLEAU:...] and [IMAGE:...] elements
        import re
        pattern = r'\[(TABLEAU|IMAGE|LOGO|SIGNATURE|CACHET):[^\]]+\]'
        
        for match in re.finditer(pattern, page_content):
            protected_elements.append({
                'start': match.start(),
                'end': match.end(),
                'text': match.group()
            })
        
        # Create base metadata
        metadata = {
            'source_document': source_document,
            'page_number': page_number
        }
        
        # Simple approach: just chunk the text normally
        # The overlap will help maintain context even if we split elements
        chunks = self.chunk_text(page_content, metadata)
        
        # Add chunk numbering for this page
        for i, chunk in enumerate(chunks):
            chunk['paragraph_number'] = i + 1  # Compatible with existing system
        
        return chunks
    
    def process_document_results(self, unified_results: Dict) -> List[Dict]:
        """
        Process results from UnifiedDocumentProcessor into chunks.
        
        Args:
            unified_results: Results from UnifiedDocumentProcessor
            
        Returns:
            List of chunks ready for indexing
        """
        all_chunks = []
        
        for page in unified_results["pages"]:
            if page.get("status") != "success" or not page.get("content"):
                continue
            
            # Chunk the page content
            page_chunks = self.chunk_page_content(
                page_content=page["content"],
                page_number=page["page_number"],
                source_document=unified_results["document_name"]
            )
            
            # Add page type info
            for chunk in page_chunks:
                chunk['page_type'] = page.get("type", "unknown")
                chunk['processing_method'] = page.get("processing_method", "unknown")
            
            all_chunks.extend(page_chunks)
        
        return all_chunks