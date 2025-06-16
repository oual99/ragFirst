# src/pdf_analyzer.py
import pdfplumber
from typing import Dict, List, Tuple, Any
import numpy as np
from PIL import Image
import io

class PDFAnalyzer:
    def __init__(self, min_text_threshold: int = 50):
        """
        Initialize PDF analyzer.
        
        Args:
            min_text_threshold: Minimum characters to consider page as native (not scanned)
        """
        self.min_text_threshold = min_text_threshold
    
    def classify_page(self, page: Any) -> Dict:  # Using Any type for now
        """
        Classify a single page as scanned or native.
        
        Args:
            page: pdfplumber page object
            
        Returns:
            Dict with classification results
        """
        # Extract text from page
        text = page.extract_text() or ""
        text_length = len(text.strip())
        
        # Extract basic page info
        page_info = {
            "page_number": page.page_number,
            "width": page.width,
            "height": page.height,
            "text_length": text_length,
            "has_images": len(page.images) > 0,
            "has_chars": len(page.chars) > 0 if hasattr(page, 'chars') else False
        }
        
        # Classification logic
        if text_length < self.min_text_threshold:
            # Very little or no text - likely scanned
            page_info["classification"] = "scanned"
            page_info["confidence"] = 0.9 if text_length == 0 else 0.7
            page_info["reason"] = f"Text length ({text_length}) below threshold ({self.min_text_threshold})"
        else:
            # Check additional indicators for native PDF
            # Native PDFs usually have character-level information
            if hasattr(page, 'chars') and len(page.chars) > 0:
                page_info["classification"] = "native"
                page_info["confidence"] = 0.95
                page_info["reason"] = "Has character-level information"
            else:
                # Has text but no char info - might be OCR'd
                page_info["classification"] = "native"  # Treat as native but with lower confidence
                page_info["confidence"] = 0.6
                page_info["reason"] = "Has text but uncertain if OCR or native"
        
        return page_info
    
    def analyze_document(self, pdf_path: str) -> List[Dict]:
        """
        Analyze entire PDF document page by page.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of page classification results
        """
        results = []
        
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            
            for page_num, page in enumerate(pdf.pages, 1):
                print(f"Analyzing page {page_num}/{total_pages}...")
                
                try:
                    page_result = self.classify_page(page)
                    results.append(page_result)
                except Exception as e:
                    # Handle any errors in page processing
                    results.append({
                        "page_number": page_num,
                        "classification": "error",
                        "error": str(e),
                        "confidence": 0.0
                    })
        
        return results
    
    def get_page_as_image(self, pdf_path: str, page_number: int, dpi: int = 200) -> Image.Image:
        """
        Convert a PDF page to PIL Image for further processing.
        
        Args:
            pdf_path: Path to PDF file
            page_number: Page number (1-indexed)
            dpi: Resolution for image conversion
            
        Returns:
            PIL Image object
        """
        with pdfplumber.open(pdf_path) as pdf:
            page = pdf.pages[page_number - 1]
            # Convert page to image
            # Note: This requires pdf2image or similar functionality
            # For now, we'll use pdfplumber's display functionality if available
            # In production, you might want to use pdf2image library
            
            # This is a placeholder - actual implementation would use pdf2image
            # or similar library to convert PDF page to image
            pass