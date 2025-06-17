# src/pdf_page_numberer.py
import fitz  # PyMuPDF
from pathlib import Path
import os
from typing import Optional, List, Dict


class PDFPageNumberer:
    def __init__(self):
        """Initialize PDF page numberer with better visibility settings."""
        self.font_size = 11
        self.font_color = (0, 0, 0)  # Black text
        self.bg_color = (1, 1, 1)  # White background
        self.bg_opacity = 0.9  # Slightly transparent
        self.padding = 5  # Padding around text
        self.margin_bottom = 15  # Distance from bottom
    
    def add_page_numbers(self, input_path: str, output_path: Optional[str] = None) -> str:
        """
        Add page numbers to a PDF file with background for visibility.
        
        Args:
            input_path: Path to input PDF
            output_path: Path for output PDF (if None, auto-generate)
            
        Returns:
            Path to the numbered PDF
        """
        # Generate output path if not provided
        if not output_path:
            input_file = Path(input_path)
            output_dir = input_file.parent / "numbered"
            output_dir.mkdir(exist_ok=True)
            output_path = str(output_dir / f"{input_file.stem}_numbered.pdf")
        
        # Open the PDF
        doc = fitz.open(input_path)
        total_pages = len(doc)
        filename = os.path.basename(input_path)
        
        # Process each page
        for page_num, page in enumerate(doc, 1):
            # Get page dimensions
            rect = page.rect
            
            # Create the text
            text = f"{filename} - Page {page_num}/{total_pages}"
            
            # Calculate text dimensions
            text_width = fitz.get_text_length(text, fontsize=self.font_size)
            text_height = self.font_size + 2
            
            # Calculate background rectangle dimensions
            bg_width = text_width + (2 * self.padding)
            bg_height = text_height + (2 * self.padding)
            
            # Calculate position (bottom center)
            x_center = rect.width / 2
            y_bottom = rect.height - self.margin_bottom
            
            # Background rectangle coordinates
            bg_x0 = x_center - (bg_width / 2)
            bg_y0 = y_bottom - bg_height
            bg_x1 = x_center + (bg_width / 2)
            bg_y1 = y_bottom
            
            # Create background rectangle
            bg_rect = fitz.Rect(bg_x0, bg_y0, bg_x1, bg_y1)
            
            # Draw semi-transparent white background
            shape = page.new_shape()
            shape.draw_rect(bg_rect)
            shape.finish(
                fill=self.bg_color,
                fill_opacity=self.bg_opacity,
                stroke_opacity=0  # No border
            )
            shape.commit()
            
            # Calculate text position (centered in background)
            text_x = x_center - (text_width / 2)
            text_y = y_bottom - self.padding - 2  # Adjust for baseline
            
            # Add text on top of background
            page.insert_text(
                (text_x, text_y),
                text,
                fontsize=self.font_size,
                color=self.font_color,
                overlay=True
            )
        
        # Save the modified PDF
        doc.save(output_path, garbage=3, deflate=True)  # Optimize file size
        doc.close()
        
        print(f"✅ Page numbers added: {output_path}")
        return output_path
    
    def add_page_numbers_batch(self, pdf_paths: List[str], output_dir: Optional[str] = None) -> Dict[str, str]:
        """
        Add page numbers to multiple PDFs.
        
        Args:
            pdf_paths: List of PDF file paths
            output_dir: Directory for output PDFs
            
        Returns:
            Dict mapping original paths to numbered paths
        """
        results = {}
        
        for pdf_path in pdf_paths:
            try:
                numbered_path = self.add_page_numbers(pdf_path)
                results[pdf_path] = numbered_path
            except Exception as e:
                print(f"❌ Error processing {pdf_path}: {str(e)}")
                results[pdf_path] = None
        
        return results