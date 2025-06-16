# src/vision_processor.py
import base64
from typing import Dict, List, Optional
from openai import OpenAI
from PIL import Image
import io
import pdf2image
import os

class VisionProcessor:
    def __init__(self, api_key: str = None, max_image_size: tuple = (1024, 1024)):
        """
        Initialize Vision Processor for handling scanned pages.
        
        Args:
            api_key: OpenAI API key
            max_image_size: Maximum image dimensions (width, height) to send to API
        """
        self.client = OpenAI(api_key=api_key)
        self.max_image_size = max_image_size
    
    def pdf_page_to_image(self, pdf_path: str, page_number: int, dpi: int = 200) -> Image.Image:
        """
        Convert a PDF page to PIL Image.
        
        Args:
            pdf_path: Path to PDF file
            page_number: Page number (1-indexed)
            dpi: Resolution for conversion
            
        Returns:
            PIL Image object
        """
        # Convert specific page to image
        images = pdf2image.convert_from_path(
            pdf_path,
            first_page=page_number,
            last_page=page_number,
            dpi=dpi
        )
        return images[0] if images else None
    
    def resize_image(self, image: Image.Image) -> Image.Image:
        """
        Resize image to fit within max dimensions while maintaining aspect ratio.
        
        Args:
            image: PIL Image object
            
        Returns:
            Resized PIL Image object
        """
        # Calculate the scaling factor
        width_ratio = self.max_image_size[0] / image.width
        height_ratio = self.max_image_size[1] / image.height
        scale_factor = min(width_ratio, height_ratio, 1.0)  # Don't upscale
        
        if scale_factor < 1.0:
            new_width = int(image.width * scale_factor)
            new_height = int(image.height * scale_factor)
            return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        return image
    
    def image_to_base64(self, image: Image.Image) -> str:
        """
        Convert PIL Image to base64 string.
        
        Args:
            image: PIL Image object
            
        Returns:
            Base64 encoded string
        """
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    def process_scanned_page(self, 
                        pdf_path: str, 
                        page_number: int,
                        document_context: str = "",
                        language: str = "French") -> Dict:
        """
        Process a scanned page using OpenAI Vision API.
        """
        # Convert PDF page to image
        image = self.pdf_page_to_image(pdf_path, page_number)
        if not image:
            return {"error": "Failed to convert PDF page to image"}
        
        # Resize if needed
        image = self.resize_image(image)
        
        # Convert to base64
        image_base64 = self.image_to_base64(image)
        
        # Even more specific prompt for signatures
        prompt = f"""Analyze this page from a construction/building (BTP) document. 
    {f'Document context: {document_context}' if document_context else ''}

    IMPORTANT: The document is in French. Extract all content in French.

    Extract ALL content from this page in a single, coherent text flow, maintaining the exact reading order.

    PAY SPECIAL ATTENTION TO:
    1. Handwritten signatures (usually blue or black ink, cursive writing)
    2. Company stamps/cachets (circular or rectangular stamps with company info)
    3. Official seals
    4. Handwritten annotations
    5. Logos and letterheads

    FORMAT FOR EACH ELEMENT:
    - Company logos/letterheads: [LOGO: description]
    - Handwritten signatures: [SIGNATURE: description, e.g., "Signature manuscrite de M. Petit Nicolas"]
    - Company stamps: [CACHET: description, e.g., "Cachet rond de l'entreprise SICRA avec date"]
    - Tables: [TABLEAU: detailed description]
    - Checkboxes: [X] for checked, [ ] for unchecked

    IMPORTANT: After text like "Le titulaire" or "signature", there is usually a signature and/or stamp. Make sure to describe them.

    Example of a complete extraction with signature:
    "Fait à Paris, le 01/01/2024
    Le Directeur
    [SIGNATURE: Signature manuscrite en encre bleue, illisible]
    [CACHET: Cachet rond de l'entreprise avec numéro SIRET]"

    Extract everything you see, in order."""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_base64}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=4000,
                temperature=0.1
            )
            
            content = response.choices[0].message.content.strip()
            
            # Extract elements with updated patterns
            tables = self._extract_elements(content, "TABLEAU")
            signatures = self._extract_elements(content, "SIGNATURE")
            stamps = self._extract_elements(content, "CACHET")
            logos = self._extract_elements(content, "LOGO")
            
            # Combine all image-type elements
            all_visual_elements = signatures + stamps + logos
            
            return {
                "page_number": page_number,
                "status": "success",
                "text": content,
                "tables": tables,
                "signatures": signatures,  # Separate tracking for signatures
                "stamps": stamps,  # Separate tracking for stamps
                "logos": logos,  # Separate tracking for logos
                "images": all_visual_elements,  # All visual elements combined
                "processing_method": "openai_vision",
                "image_dimensions": {"width": image.width, "height": image.height}
            }
            
        except Exception as e:
            return {
                "page_number": page_number,
                "status": "error",
                "error": str(e),
                "processing_method": "openai_vision"
            }
    def _extract_elements(self, text: str, element_type: str) -> List[Dict]:
        """
        Extract element descriptions from the text.
        
        Args:
            text: The full extracted text
            element_type: Regex pattern for element type (e.g., "TABLEAU", "IMAGE|DIAGRAMME")
            
        Returns:
            List of element descriptions with their positions
        """
        import re
        
        elements = []
        pattern = rf'\[({element_type}): ([^\]]+)\]'
        
        for match in re.finditer(pattern, text):
            elements.append({
                "type": match.group(1),
                "description": match.group(2),
                "position": match.start()
            })
        
        return elements

    def _parse_vision_response(self, response: str) -> Dict:
        """
        Parse the structured response from Vision API.
        Updated to handle French section markers.
        """
        sections = {
            "text_content": "",
            "elements_analysis": "",
            "key_information": ""
        }
        
        # Handle both English and French section markers
        text_markers = ["=== CONTENU TEXTUEL ===", "=== TEXT CONTENT ==="]
        elements_markers = ["=== ANALYSE DES ÉLÉMENTS ===", "=== ELEMENTS ANALYSIS ==="]
        info_markers = ["=== INFORMATIONS CLÉS ===", "=== KEY INFORMATION ==="]
        
        # Find which markers are used
        text_marker = next((m for m in text_markers if m in response), None)
        elements_marker = next((m for m in elements_markers if m in response), None)
        info_marker = next((m for m in info_markers if m in response), None)
        
        if text_marker:
            parts = response.split(text_marker)[1]
            if elements_marker and elements_marker in parts:
                sections["text_content"] = parts.split(elements_marker)[0].strip()
                remainder = parts.split(elements_marker)[1]
                if info_marker and info_marker in remainder:
                    sections["elements_analysis"] = remainder.split(info_marker)[0].strip()
                    sections["key_information"] = remainder.split(info_marker)[1].strip()
        
        return sections