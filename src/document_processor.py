"""Document processing module for PDF extraction."""
import os
from typing import List, Dict
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import NarrativeText


class DocumentProcessor:
    def __init__(self, images_output_dir: str = "./data/images/"):
        self.images_output_dir = images_output_dir
        os.makedirs(self.images_output_dir, exist_ok=True)
    
    def process_pdf(self, document_path: str) -> List:
        """Process a PDF document and extract its contents."""
        return partition_pdf(
            filename=document_path,
            extract_images_in_pdf=True,
            extract_image_block_to_payload=False,
            extract_image_block_output_dir=self.images_output_dir
        )
    
    def extract_text_with_metadata(self, 
                                   processed_document: List, 
                                   source_document: str) -> List[Dict]:
        """Extract text with metadata from processed document."""
        text_data = []
        paragraph_counters = {}
        
        for element in processed_document:
            if isinstance(element, NarrativeText):
                page_number = element.metadata.page_number
                
                if page_number not in paragraph_counters:
                    paragraph_counters[page_number] = 1
                else:
                    paragraph_counters[page_number] += 1
                
                paragraph_number = paragraph_counters[page_number]
                
                text_content = element.text
                text_data.append({
                    "source_document": source_document,
                    "page_number": page_number,
                    "paragraph_number": paragraph_number,
                    "text": text_content
                })
        
        return text_data