# test_vision_processor.py
from src.vision_processor import VisionProcessor
from src.pdf_analyzer import PDFAnalyzer
import os
import config
import time

def test_vision_processing(pdf_path: str, test_page: int = None):
    """Test vision processing on scanned pages."""
    
    # First, analyze the PDF to find scanned pages
    analyzer = PDFAnalyzer()
    print(f"Analyzing PDF structure: {pdf_path}")
    
    start_time = time.time()
    page_results = analyzer.analyze_document(pdf_path)
    analysis_time = time.time() - start_time
    
    print(f"PDF analysis completed in {analysis_time:.2f} seconds")
    
    # Find scanned pages
    scanned_pages = [r for r in page_results if r.get("classification") == "scanned"]
    
    if not scanned_pages:
        print("No scanned pages found in this document!")
        return
    
    print(f"\nFound {len(scanned_pages)} scanned pages")
    
    # Initialize vision processor
    vision_processor = VisionProcessor(api_key=config.OPENAI_API_KEY)
    
    # Process a specific page or the first scanned page
    if test_page:
        pages_to_process = [p for p in scanned_pages if p["page_number"] == test_page]
    else:
        pages_to_process = [scanned_pages[0]]  # Just test the first scanned page
    
    # Track total processing time
    total_processing_time = 0
    
    for page_info in pages_to_process:
        page_num = page_info["page_number"]
        print(f"\n{'='*60}")
        print(f"Processing scanned page {page_num}")
        print(f"{'='*60}")
        
        # Time the processing
        page_start_time = time.time()
        
        result = vision_processor.process_scanned_page(
            pdf_path=pdf_path,
            page_number=page_num,
            document_context="Construction/BTP document",
            language="French"
        )
        
        page_processing_time = time.time() - page_start_time
        total_processing_time += page_processing_time
        
        print(f"\n⏱️ Page {page_num} processed in {page_processing_time:.2f} seconds")
        
        if result["status"] == "success":
            print("\n--- Extracted Text ---")
            # print(result)
            text = result.get("text", "")
            # # print(text[:1000] + "..." if len(text) > 1000 else text)
            print(text)
            
            # # Show found elements
            # if result.get("tables"):
            #     print(f"\n--- Found {len(result['tables'])} Tables ---")
            #     for i, table in enumerate(result["tables"][:3]):  # Show first 3
            #         print(f"Table {i+1}: {table['description'][:100]}...")
            
            # if result.get("images"):
            #     print(f"\n--- Found {len(result['images'])} Images/Diagrams ---")
            #     for i, img in enumerate(result["images"][:3]):  # Show first 3
            #         print(f"Image {i+1}: {img['description'][:100]}...")
                    
            # # Show performance metrics
            # print(f"\n--- Performance Metrics ---")
            # print(f"Text length: {len(text)} characters")
            # print(f"Processing speed: {len(text) / page_processing_time:.0f} chars/second")
                    
        else:
            print(f"Error: {result.get('error', 'Unknown error')}")
    
    # Summary statistics
    print(f"\n{'='*60}")
    print("PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"Total pages processed: {len(pages_to_process)}")
    print(f"Total processing time: {total_processing_time:.2f} seconds")
    if len(pages_to_process) > 0:
        print(f"Average time per page: {total_processing_time / len(pages_to_process):.2f} seconds")

if __name__ == "__main__":
    # Test with your PDF
    test_vision_processing("docs_test/2.1.Acte_d_Engagement_signé_01_10_2018.pdf", test_page=1)