# src/document_processor.py 
"""Enhanced document processing module with unified PDF extraction."""
import os
from typing import List, Dict
from .unified_document_processor import UnifiedDocumentProcessor
from .simple_chunker import SimpleChunker


class DocumentProcessor:
    def __init__(self, openai_api_key: str = None):
        """
        Initialize the enhanced document processor.
        
        Args:
            openai_api_key: OpenAI API key for vision processing
        """
        import config
        self.openai_api_key = openai_api_key or config.OPENAI_API_KEY
        self.unified_processor = UnifiedDocumentProcessor(self.openai_api_key)
        self.chunker = SimpleChunker(chunk_size=500, overlap_size=100)
    
    def process_pdf(self, document_path: str, progress_callback=None) -> Dict:
        """
        Process a PDF document using the unified processor.
        
        Args:
            document_path: Path to PDF file
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dict with processing results
        """
        return self.unified_processor.process_document(
            pdf_path=document_path,
            progress_callback=progress_callback,
            describe_images=True
        )
    
    def extract_text_with_metadata(self, 
                                   processed_document: Dict, 
                                   source_document: str) -> List[Dict]:
        """
        Extract text chunks with metadata from processed document.
        
        Args:
            processed_document: Output from process_pdf
            source_document: Source document path
            
        Returns:
            List of text chunks ready for indexing
        """
        # Use the simple chunker to process results
        chunks = self.chunker.process_document_results(processed_document)
        
        # Ensure source document name is consistent
        doc_name = os.path.basename(source_document)
        for chunk in chunks:
            chunk['source_document'] = doc_name
        
        return chunks