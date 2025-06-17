# test_native_processor.py
import sys
import os
import time
import warnings
import logging
from contextlib import redirect_stderr
from src.native_pdf_processor import NativePDFProcessor
from src.pdf_analyzer import PDFAnalyzer
import config

# Suppress PyMuPDF warnings globally
logging.getLogger("fitz").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

def test_native_processing(pdf_path: str, test_page: int = None):
    """Test native PDF processing."""
    
    # Suppress warnings during PDF analysis
    with open(os.devnull, 'w') as devnull:
        with redirect_stderr(devnull):
            # First, find native pages
            analyzer = PDFAnalyzer()
            print(f"Analyzing PDF structure: {pdf_path}")
            page_results = analyzer.analyze_document(pdf_path)
    
    # Find native pages
    native_pages = [r for r in page_results if r.get("classification") == "native"]
    
    if not native_pages:
        print("No native pages found!")
        return
    
    print(f"\nFound {len(native_pages)} native pages")
    
    # Initialize processor
    processor = NativePDFProcessor(api_key=config.OPENAI_API_KEY)
    
    # Process specific page or first native page
    if test_page:
        pages_to_process = [p for p in native_pages if p["page_number"] == test_page]
    else:
        pages_to_process = [native_pages[0]]
    
    total_processing_time = 0
    
    for page_info in pages_to_process:
        page_num = page_info["page_number"]
        print(f"\n{'='*60}")
        print(f"Processing native page {page_num}")
        print(f"{'='*60}")
        
        print(page_info)
        start_time = time.time()
        
        # Suppress warnings during page processing
        with open(os.devnull, 'w') as devnull:
            with redirect_stderr(devnull):
                result = processor.process_native_page(
                    pdf_path=pdf_path,
                    page_number=page_num,
                    describe_images=True
                )
        
        processing_time = time.time() - start_time
        total_processing_time += processing_time
        
        print(f"\n⏱️ Page processed in {processing_time:.2f} seconds")
        
        if result["status"] == "success":
            print(f"\nText blocks found: {len(result['text_blocks'])}")
            print(f"Tables found: {len(result['tables'])}")
            print(f"Images found: {len(result['images'])}")
            
            print("\n--- Merged Content Preview ---")
            content = result["merged_content"]
            # print(content[:1000] + "..." if len(content) > 1000 else content)
            print(content)
            
            # # Debug table positions
            # if result["tables"]:
            #     print("\n--- Table Positions Debug ---")
            #     for i, table in enumerate(result["tables"]):
            #         print(f"Table {i+1}: Position Y0={table.get('y0', 'N/A')}, Y1={table.get('y1', 'N/A')}")
            #         print(f"  First row: {table['rows'][0] if table['rows'] else 'N/A'}")


            # if result["tables"]:
            #     print(f"\n--- Tables Details ---")
            #     for i, table in enumerate(result["tables"]):
            #         print(f"\nTable {i+1}:")
            #         print(f"  Size: {table['num_rows']}x{table['num_cols']}")
            #         print(f"  Description: {table['description']}")
            #         # Show first few rows if you want
            #         if table['rows'] and len(table['rows']) > 0:
            #             print(f"  Headers: {table['rows'][0] if table['rows'] else 'N/A'}")
            
            # # Show image descriptions
            # if result["images"]:
            #     print(f"\n--- Images Details ---")
            #     for i, img in enumerate(result["images"]):
            #         print(f"\nImage {i+1}:")
            #         print(f"  Size: {img['width']}x{img['height']}px")
            #         print(f"  Position: ({img['x0']:.1f}, {img['y0']:.1f})")
            #         desc = img.get('description', 'No description')
            #         print(f"  Description: {desc[:200]}..." if len(desc) > 200 else f"  Description: {desc}")
            
            # Show performance metrics
            print(f"\n--- Performance Metrics ---")
            print(f"Total content length: {len(result['merged_content'])} characters")
            print(f"Processing speed: {len(result['merged_content']) / processing_time:.0f} chars/second")
            
        else:
            print(f"Error: {result.get('error', 'Unknown error')}")
    
    # Summary statistics
    if len(pages_to_process) > 1:
        print(f"\n{'='*60}")
        print("PROCESSING SUMMARY")
        print(f"{'='*60}")
        print(f"Total pages processed: {len(pages_to_process)}")
        print(f"Total processing time: {total_processing_time:.2f} seconds")
        print(f"Average time per page: {total_processing_time / len(pages_to_process):.2f} seconds")

if __name__ == "__main__":
    # Test with your PDF - adjust the path as needed
    test_native_processing("docs_test/Fiche Synthese RT2012 - 102lgts Av Verdun ISSY LES MLX - CODIBAT du 100918.pdf", test_page=3)