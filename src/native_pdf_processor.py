# src/native_pdf_processor.py
import pdfplumber
from typing import Dict, List, Optional, Tuple
import base64
from PIL import Image
import io
from openai import OpenAI
import fitz  # PyMuPDF for image extraction
import warnings
import logging
from contextlib import redirect_stderr

class NativePDFProcessor:
    def __init__(self, api_key: str = None):
        """
        Initialize processor for native PDF content.
        
        Args:
            api_key: OpenAI API key for image/diagram description
        """
        self.client = OpenAI(api_key=api_key) if api_key else None
    
    def process_native_page(self, 
                        pdf_path: str, 
                        page_number: int,
                        describe_images: bool = True) -> Dict:
        """Process a native PDF page extracting all elements."""
        result = {
            "page_number": page_number,
            "status": "success",
            "text_blocks": [],
            "tables": [],
            "images": [],
            "merged_content": ""
        }
        
        try:
            # Process with pdfplumber for text and tables
            with pdfplumber.open(pdf_path) as pdf:
                page = pdf.pages[page_number - 1]
                
                # Extract text blocks with positions
                text_blocks = self._extract_text_blocks(page)
                result["text_blocks"] = text_blocks
                
                # Extract tables with custom settings to reduce false positives
                # Use stricter table detection settings
                custom_table_settings = {
                    "vertical_strategy": "lines", 
                    "horizontal_strategy": "lines",
                    "explicit_vertical_lines": [],
                    "explicit_horizontal_lines": [],
                    "snap_tolerance": 3,
                    "join_tolerance": 3,
                    "edge_min_length": 3,
                    "min_words_vertical": 3,
                    "min_words_horizontal": 2,
                    "text_tolerance": 3,
                    "intersection_tolerance": 3,
                }
                
                # Find tables with custom settings
                tables = page.find_tables(table_settings=custom_table_settings)
                
                # Process found tables
                extracted_tables = []
                for table in tables:
                    if table:
                        extracted = table.extract()
                        if extracted and len(extracted) > 1:  # Must have more than one row
                            num_cols = max(len(row) for row in extracted)
                            # Filter out false positives
                            # Skip single column with 2 rows or less (likely titles)
                            if num_cols == 1 and len(extracted) <= 2:
                                continue
                            
                            bbox = table.bbox
                            table_data = {
                                'rows': extracted,
                                'num_rows': len(extracted),
                                'num_cols': num_cols,
                                'html': self._table_to_html(extracted),
                                'description': self._generate_table_description(extracted),
                                'x0': bbox[0],
                                'y0': bbox[1],
                                'x1': bbox[2],
                                'y1': bbox[3]
                            }
                            extracted_tables.append(table_data)
                
                result["tables"] = extracted_tables
        
            # Rest of the method remains the same...
            # Process with PyMuPDF for images
            doc = fitz.open(pdf_path)
            pdf_page = doc[page_number - 1]
            
            # Extract images
            images = self._extract_images(pdf_page, pdf_path, page_number)
            
            # If no images found, try alternative method
            if not images:
                images = self._extract_images_alternative(pdf_page, page_number)
            
            # Describe images if requested
            if describe_images and images and self.client:
                images = self._describe_images(images)
            
            result["images"] = images
            doc.close()
            
            # Merge all content in reading order
            result["merged_content"] = self._merge_content_in_order(
                text_blocks, result["tables"], images, page
            )
            
        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
        
        return result
    
    def _extract_text_blocks(self, page) -> List[Dict]:
        """Extract text blocks with their positions."""
        text_blocks = []
        
        # Get text with bounding boxes
        words = page.extract_words(
            x_tolerance=3,
            y_tolerance=3,
            keep_blank_chars=True
        )
        
        if not words:
            return []
        
        # Group words into lines
        lines = []
        current_line = [words[0]]
        
        for word in words[1:]:
            # Check if word is on the same line (similar y-coordinate)
            if abs(word['top'] - current_line[-1]['top']) < 5:
                current_line.append(word)
            else:
                lines.append(current_line)
                current_line = [word]
        
        if current_line:
            lines.append(current_line)
        
        # Convert lines to text blocks
        for line in lines:
            text = ' '.join([w['text'] for w in line])
            text_blocks.append({
                'text': text,
                'x0': min(w['x0'] for w in line),
                'y0': min(w['top'] for w in line),
                'x1': max(w['x1'] for w in line),
                'y1': max(w['bottom'] for w in line)
            })
        
        return text_blocks
    
    def _extract_tables(self, page) -> List[Dict]:
        """Extract tables from the page with their positions."""
        tables_data = []
        
        # First, find table locations using find_tables
        table_finder = page.find_tables()
        
        for i, table in enumerate(table_finder):
            if not table:
                continue
                
            # Extract the table data
            extracted = table.extract()
            
            if not extracted or not any(extracted):
                continue
            
            # Filter out false positives
            # A real table should have:
            # 1. More than one column OR more than 2 rows
            # 2. Not be just a title (single cell tables are usually titles)
            num_rows = len(extracted)
            num_cols = max(len(row) for row in extracted) if extracted else 0
            
            # Skip if it's a single cell or single column with less than 3 rows
            if num_cols == 1 and num_rows <= 2:
                # Check if it's likely just a title/header
                total_text = ' '.join([cell for row in extracted for cell in row if cell])
                # If it's short text without typical table indicators, skip it
                if len(total_text) < 100 and not any(indicator in total_text.lower() for indicator in ['total', 'somme', '€', '%', 'nombre']):
                    continue
            
            # Get table bounding box
            bbox = table.bbox  # This gives us (x0, top, x1, bottom)
            
            table_dict = {
                'table_index': i,
                'rows': extracted,
                'num_rows': num_rows,
                'num_cols': num_cols,
                'html': self._table_to_html(extracted),
                'description': self._generate_table_description(extracted),
                'x0': bbox[0],
                'y0': bbox[1],  # top position
                'x1': bbox[2],
                'y1': bbox[3]   # bottom position
            }
            
            tables_data.append(table_dict)
        
        return tables_data
    
    def _table_to_html(self, table: List[List]) -> str:
        """Convert table to HTML format."""
        if not table:
            return ""
        
        html = "<table border='1'>\n"
        
        # First row as header
        if len(table) > 0:
            html += "  <thead>\n    <tr>\n"
            for cell in table[0]:
                html += f"      <th>{cell if cell else ''}</th>\n"
            html += "    </tr>\n  </thead>\n"
        
        # Rest as body
        if len(table) > 1:
            html += "  <tbody>\n"
            for row in table[1:]:
                html += "    <tr>\n"
                for cell in row:
                    html += f"      <td>{cell if cell else ''}</td>\n"
                html += "    </tr>\n"
            html += "  </tbody>\n"
        
        html += "</table>"
        return html
    
    def _generate_table_description(self, table: List[List]) -> str:
        """Generate a text description of the table."""
        if not table:
            return "Tableau vide"
        
        num_rows = len(table)
        num_cols = max(len(row) for row in table) if table else 0
        
        description = f"Tableau avec {num_rows} lignes et {num_cols} colonnes. "
        
        # Describe headers
        if table and table[0]:
            headers = [str(cell) for cell in table[0] if cell]
            if headers:
                description += f"Colonnes: {', '.join(headers[:5])}{'...' if len(headers) > 5 else ''}. "
        
        # Describe content type
        if num_rows > 1:
            # Sample some cells to determine content type
            sample_cells = []
            for row in table[1:min(4, num_rows)]:
                sample_cells.extend([str(cell) for cell in row if cell])
            
            has_numbers = any(any(c.isdigit() for c in cell) for cell in sample_cells)
            has_currency = any('€' in cell or '€' in cell for cell in sample_cells)
            
            if has_currency:
                description += "Contient des montants financiers. "
            elif has_numbers:
                description += "Contient des données numériques. "
        
        return description
    
    def _extract_images(self, page, pdf_path: str, page_number: int) -> List[Dict]:
        """Extract images from the page using PyMuPDF."""
        images = []
        
        try:
            # Method 1: Try to get images using get_images()
            image_list = page.get_images(full=True)
            
            for img_index, img in enumerate(image_list):
                try:
                    # Get the XREF of the image
                    xref = img[0]
                    
                    # Extract the image
                    base_image = page.parent.extract_image(xref)
                    if not base_image:
                        continue
                        
                    # Get image data
                    image_bytes = base_image["image"]
                    
                    # Convert to PIL Image
                    pil_image = Image.open(io.BytesIO(image_bytes))
                    
                    # Get image position on page - fixed method call
                    img_instances = page.get_image_info()
                    
                    # Find the instance that matches our xref
                    bbox = None
                    for inst in img_instances:
                        if inst.get("xref") == xref:
                            bbox = inst["bbox"]
                            break
                    
                    if bbox:
                        x0, y0, x1, y1 = bbox
                    else:
                        # Try to find any bbox for this image
                        if img_instances:
                            bbox = img_instances[img_index]["bbox"] if img_index < len(img_instances) else img_instances[0]["bbox"]
                            x0, y0, x1, y1 = bbox
                        else:
                            # Fallback
                            x0, y0, x1, y1 = 0, 0, 100, 100
                    
                    images.append({
                        'image_index': img_index,
                        'image': pil_image,
                        'x0': x0,
                        'y0': y0,
                        'x1': x1,
                        'y1': y1,
                        'width': pil_image.width,
                        'height': pil_image.height,
                        'extension': base_image.get("ext", "png")
                    })
                    
                except Exception as e:
                    # Only print error if it's not the expected one
                    if "get_image_info() got an unexpected keyword argument" not in str(e):
                        print(f"Error with method 1 for image {img_index}: {str(e)}")
                    continue
            
            # Method 2: If no images found, try alternative approach
            if not images:
                img_list = page.get_image_info()
                
                for img_index, img_info in enumerate(img_list):
                    try:
                        bbox = img_info["bbox"]
                        mat = fitz.Matrix(2, 2)
                        rect = fitz.Rect(bbox)
                        pix = page.get_pixmap(matrix=mat, clip=rect)
                        
                        img_data = pix.tobytes("png")
                        pil_image = Image.open(io.BytesIO(img_data))
                        
                        images.append({
                            'image_index': img_index,
                            'image': pil_image,
                            'x0': bbox[0],
                            'y0': bbox[1],
                            'x1': bbox[2],
                            'y1': bbox[3],
                            'width': pil_image.width,
                            'height': pil_image.height,
                            'method': 'pixmap_extraction'
                        })
                        
                    except Exception as e:
                        continue
                        
        except Exception as e:
            print(f"General error in image extraction: {str(e)}")
        
        return images
    
    def _extract_images_alternative(self, page, page_number: int) -> List[Dict]:
        """Alternative method to extract images by rendering the page."""
        images = []
        
        try:
            # Render the page at high resolution
            mat = fitz.Matrix(2, 2)  # 2x zoom
            pix = page.get_pixmap(matrix=mat)
            
            # Convert to PIL Image for analysis
            img_data = pix.tobytes("png")
            page_image = Image.open(io.BytesIO(img_data))
            
            # Now we need to detect image regions
            # This is a simplified approach - in production you might use
            # computer vision techniques to detect image regions
            
            # For now, let's check if the page has any embedded images
            # and try to extract their regions
            image_info = page.get_image_info()
            
            for idx, info in enumerate(image_info):
                bbox = info.get("bbox", None)
                if bbox:
                    # Scale bbox coordinates to match the rendered image
                    x0, y0, x1, y1 = bbox
                    x0, y0, x1, y1 = int(x0 * 2), int(y0 * 2), int(x1 * 2), int(y1 * 2)
                    
                    # Crop the region from the full page
                    cropped = page_image.crop((x0, y0, x1, y1))
                    
                    images.append({
                        'image_index': idx,
                        'image': cropped,
                        'x0': bbox[0],
                        'y0': bbox[1],
                        'x1': bbox[2],
                        'y1': bbox[3],
                        'width': cropped.width,
                        'height': cropped.height,
                        'method': 'region_extraction'
                    })
            
        except Exception as e:
            print(f"Error in alternative image extraction: {str(e)}")
        
        return images

    def _describe_images(self, images: List[Dict]) -> List[Dict]:
        """Use OpenAI to describe images."""
        for img_data in images:
            try:
                # Convert PIL image to base64
                buffered = io.BytesIO()
                img_data['image'].save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                
                # Check image size to determine if it's likely a logo
                width = img_data['width']
                height = img_data['height']
                aspect_ratio = width / height if height > 0 else 1
                
                # Likely a logo if small or has typical logo dimensions
                is_likely_logo = (width < 300 and height < 300) or (aspect_ratio > 2.5 or aspect_ratio < 0.4)
                
                # Different prompts based on image type
                if is_likely_logo:
                    prompt = """Décris brièvement ce logo ou cette image. 
                    Sois factuel et concis. 
                    Si du texte est visible, transcris-le exactement.
                    Maximum 2-3 phrases.
                    
                    Exemple: "Logo avec une maison stylisée en jaune et bleu, texte 'AB PROGRAMME' en dessous." """
                else:
                    prompt = """Décris cette image/diagramme d'un document BTP en français. 
                    Sois précis et technique. 
                    Si c'est un diagramme ou schéma, décris tous les éléments, labels et relations.
                    Si c'est une photo, décris ce qui est montré."""
                
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
                                        "url": f"data:image/png;base64,{img_base64}",
                                        "detail": "high" if not is_likely_logo else "low"
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=500 if not is_likely_logo else 150,
                    temperature=0.1
                )
                
                description = response.choices[0].message.content
                
                # Add image type to the data
                img_data['description'] = description
                img_data['image_type'] = 'logo' if is_likely_logo else 'image'
                
            except Exception as e:
                img_data['description'] = f"Erreur lors de la description: {str(e)}"
                img_data['image_type'] = 'unknown'
        
        return images
    
    def _merge_content_in_order(self, text_blocks: List[Dict], 
                           tables: List[Dict], 
                           images: List[Dict],
                           page) -> str:
        """Merge all content in reading order based on vertical position."""
        # Create a list of all elements with their positions
        elements = []
        
        # Add text blocks
        for block in text_blocks:
            elements.append({
                'type': 'text',
                'content': block['text'],
                'y0': block['y0'],
                'y1': block.get('y1', block['y0'] + 10)
            })
        
        # Add tables with their actual positions
        for table in tables:
            elements.append({
                'type': 'table',
                'content': f"[TABLEAU: {table['description']}]",
                'y0': table.get('y0', 0),
                'y1': table.get('y1', table.get('y0', 0) + 50)
            })
        
        # Add images
        for img in images:
            if img.get('image_type') == 'logo':
                content = f"[LOGO: {img.get('description', 'Logo sans description')}]"
            else:
                content = f"[IMAGE: {img.get('description', 'Image sans description')}]"
            
            elements.append({
                'type': 'image',
                'content': content,
                'y0': img['y0'],
                'y1': img.get('y1', img['y0'] + 50)
            })
        
        # Sort by vertical position (top of element)
        elements.sort(key=lambda x: x['y0'])
        
        # Group elements that overlap vertically (like multi-column layouts)
        merged_content = []
        current_line = []
        current_y = -1
        
        for elem in elements:
            # If this element is on a different line (significantly different y position)
            if current_y == -1 or abs(elem['y0'] - current_y) > 5:
                # Add previous line if exists
                if current_line:
                    merged_content.append(' '.join([e['content'] for e in current_line]))
                current_line = [elem]
                current_y = elem['y0']
            else:
                # Same line, add to current line
                current_line.append(elem)
        
        # Don't forget the last line
        if current_line:
            merged_content.append(' '.join([e['content'] for e in current_line]))
        
        return '\n'.join(merged_content)