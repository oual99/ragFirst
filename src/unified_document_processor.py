# src/unified_document_processor.py - Without caching
from typing import Dict, List, Optional, Callable
import os
from datetime import datetime
from .pdf_analyzer import PDFAnalyzer
from .vision_processor import VisionProcessor
from .native_pdf_processor import NativePDFProcessor
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


class UnifiedDocumentProcessor:
    def __init__(self, openai_api_key: str):
        """
        Initialize unified document processor.
        
        Args:
            openai_api_key: OpenAI API key for vision and descriptions
        """
        self.pdf_analyzer = PDFAnalyzer()
        self.vision_processor = VisionProcessor(api_key=openai_api_key)
        self.native_processor = NativePDFProcessor(api_key=openai_api_key)
    
    def _process_pages_parallel(self, pdf_path: str, page_classifications: List[Dict], 
                              describe_images: bool, progress_callback=None) -> List[Dict]:
        """Process multiple pages in parallel."""
        page_results = [None] * len(page_classifications)  # Maintain order
        total_pages = len(page_classifications)
        completed_pages = 0
        lock = threading.Lock()
        
        def update_progress(message):
            nonlocal completed_pages
            with lock:
                completed_pages += 1
                if progress_callback and completed_pages <= total_pages:
                    # Ensure progress never exceeds 0.9 (leave 0.9-1.0 for finalization)
                    progress = 0.1 + (0.8 * (min(completed_pages, total_pages) / total_pages))
                    progress = min(progress, 0.9)  # Cap at 0.9
                    progress_callback(progress, f"{message} ({completed_pages}/{total_pages})")
        
        def process_single_page(idx, page_class):
            """Process a single page and store result."""
            page_num = page_class["page_number"]
            
            try:
                # Process based on page type
                if page_class["classification"] == "scanned":
                    page_result = self._process_scanned_page(
                        pdf_path, page_num, describe_images
                    )
                elif page_class["classification"] == "native":
                    page_result = self._process_native_page(
                        pdf_path, page_num, describe_images
                    )
                else:
                    page_result = {
                        "page_number": page_num,
                        "type": "error",
                        "error": page_class.get("error", "Unknown classification"),
                        "content": ""
                    }
                
                # Add classification info
                page_result["classification"] = page_class["classification"]
                page_result["classification_confidence"] = page_class.get("confidence", 0)
                
                # Store result in correct position
                page_results[idx] = page_result
                update_progress(f"Page {page_num} traitée")
                
            except Exception as e:
                page_results[idx] = {
                    "page_number": page_num,
                    "type": "error",
                    "error": str(e),
                    "content": "",
                    "classification": page_class["classification"]
                }
                update_progress(f"Page {page_num} - erreur")
        
        # Use ThreadPoolExecutor for parallel processing
        # Limit workers to avoid overwhelming APIs
        max_workers = min(5, len(page_classifications))  # Max 5 concurrent pages
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all pages for processing
            futures = {}
            for idx, page_class in enumerate(page_classifications):
                future = executor.submit(process_single_page, idx, page_class)
                futures[future] = idx
            
            # Wait for all to complete
            for future in as_completed(futures):
                try:
                    future.result()  # This will raise any exceptions that occurred
                except Exception as e:
                    # Don't print progress errors, handle other errors
                    if "Progress Value has invalid value" not in str(e):
                        print(f"Error processing page: {e}")
        
        return page_results
    
    def process_document(self, 
                        pdf_path: str,
                        progress_callback: Optional[Callable] = None,
                        describe_images: bool = True) -> Dict:
        """
        Process entire PDF document with parallel processing.
        
        Args:
            pdf_path: Path to PDF file
            progress_callback: Optional callback for progress updates
            describe_images: Whether to use AI to describe images
            
        Returns:
            Dict with complete document processing results
        """
        start_time = datetime.now()
        
        # Initialize results structure
        results = {
            "document_path": pdf_path,
            "document_name": os.path.basename(pdf_path),
            "processing_date": start_time.isoformat(),
            "pages": [],
            "summary": {
                "total_pages": 0,
                "scanned_pages": 0,
                "native_pages": 0,
                "total_images": 0,
                "total_tables": 0,
                "processing_time": 0
            },
            "status": "success",
            "errors": []
        }
        
        try:
            # Step 1: Analyze document structure
            if progress_callback:
                progress_callback(0.1, "Analyse de la structure du document...")
            
            page_classifications = self.pdf_analyzer.analyze_document(pdf_path)
            results["summary"]["total_pages"] = len(page_classifications)
            
            # Count page types
            for page_class in page_classifications:
                if page_class.get("classification") == "scanned":
                    results["summary"]["scanned_pages"] += 1
                elif page_class.get("classification") == "native":
                    results["summary"]["native_pages"] += 1
            
            # Step 2: Process pages in parallel
            if progress_callback:
                progress_callback(0.15, f"Traitement parallèle de {len(page_classifications)} pages...")
            
            # Use parallel processing
            processed_pages = self._process_pages_parallel(
                pdf_path, 
                page_classifications, 
                describe_images,
                progress_callback
            )
            
            # Store results and update summary
            for page_result in processed_pages:
                if page_result:  # Skip None results
                    results["pages"].append(page_result)
                    
                    # Update summary
                    if "images" in page_result:
                        results["summary"]["total_images"] += len(page_result.get("images", []))
                    if "tables" in page_result:
                        results["summary"]["total_tables"] += len(page_result.get("tables", []))
            
            # Step 3: Finalize
            if progress_callback:
                progress_callback(0.95, "Finalisation...")
            
            end_time = datetime.now()
            results["summary"]["processing_time"] = (end_time - start_time).total_seconds()
            
            if progress_callback:
                progress_callback(1.0, "Traitement terminé!")
                
        except Exception as e:
            results["status"] = "error"
            results["errors"].append(str(e))
            
        return results
    
    def _process_scanned_page(self, pdf_path: str, page_num: int, describe_images: bool) -> Dict:
        """Process a scanned page using Vision API."""
        try:
            result = self.vision_processor.process_scanned_page(
                pdf_path=pdf_path,
                page_number=page_num,
                document_context=os.path.basename(pdf_path),
                language="French"
            )
            
            if result["status"] == "success":
                return {
                    "page_number": page_num,
                    "type": "scanned",
                    "content": result["text"],
                    "processing_method": "openai_vision",
                    "images": result.get("images", []),
                    "tables": result.get("tables", []),
                    "status": "success"
                }
            else:
                return {
                    "page_number": page_num,
                    "type": "scanned",
                    "content": "",
                    "error": result.get("error", "Unknown error"),
                    "status": "error"
                }
                
        except Exception as e:
            return {
                "page_number": page_num,
                "type": "scanned",
                "content": "",
                "error": str(e),
                "status": "error"
            }
    
    def _process_native_page(self, pdf_path: str, page_num: int, describe_images: bool) -> Dict:
        """Process a native PDF page."""
        try:
            result = self.native_processor.process_native_page(
                pdf_path=pdf_path,
                page_number=page_num,
                describe_images=describe_images
            )
            
            if result["status"] == "success":
                return {
                    "page_number": page_num,
                    "type": "native",
                    "content": result["merged_content"],
                    "text_blocks": result.get("text_blocks", []),
                    "tables": result.get("tables", []),
                    "images": result.get("images", []),
                    "processing_method": "native_extraction",
                    "status": "success"
                }
            else:
                return {
                    "page_number": page_num,
                    "type": "native",
                    "content": "",
                    "error": result.get("error", "Unknown error"),
                    "status": "error"
                }
                
        except Exception as e:
            return {
                "page_number": page_num,
                "type": "native",
                "content": "",
                "error": str(e),
                "status": "error"
            }
    
    def prepare_for_indexing(self, processed_document: Dict) -> List[Dict]:
        """
        Convert processed document into chunks ready for vector indexing.
        
        Args:
            processed_document: Output from process_document
            
        Returns:
            List of text chunks with metadata for indexing
        """
        chunks = []
        doc_name = processed_document["document_name"]
        
        for page in processed_document["pages"]:
            if page.get("status") != "success" or not page.get("content"):
                continue
            
            page_num = page["page_number"]
            content = page["content"]
            
            # Split content into paragraphs
            paragraphs = content.split('\n\n')
            
            for para_num, paragraph in enumerate(paragraphs, 1):
                if paragraph.strip():
                    chunk = {
                        'source_document': doc_name,
                        'page_number': page_num,
                        'paragraph_number': para_num,
                        'text': paragraph.strip(),
                        'page_type': page.get("type", "unknown"),
                        'processing_method': page.get("processing_method", "unknown")
                    }
                    
                    # Add additional metadata if available
                    if page.get("tables"):
                        chunk["has_tables"] = True
                        chunk["num_tables"] = len(page["tables"])
                    
                    if page.get("images"):
                        chunk["has_images"] = True
                        chunk["num_images"] = len(page["images"])
                    
                    chunks.append(chunk)
        
        return chunks
    
    def save_processing_results(self, results: Dict, output_path: str):
        """Save processing results to JSON file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)