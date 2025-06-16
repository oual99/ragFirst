# test_pdf_analyzer.py
from src.pdf_analyzer import PDFAnalyzer
import json

def test_pdf_classification(pdf_path: str):
    """Test PDF page classification."""
    analyzer = PDFAnalyzer(min_text_threshold=50)
    
    print(f"Analyzing PDF: {pdf_path}")
    print("-" * 50)
    
    results = analyzer.analyze_document(pdf_path)
    
    # Summary statistics
    total_pages = len(results)
    scanned_pages = sum(1 for r in results if r.get("classification") == "scanned")
    native_pages = sum(1 for r in results if r.get("classification") == "native")
    error_pages = sum(1 for r in results if r.get("classification") == "error")
    
    print(f"\nSummary:")
    print(f"Total pages: {total_pages}")
    print(f"Scanned pages: {scanned_pages}")
    print(f"Native pages: {native_pages}")
    print(f"Error pages: {error_pages}")
    
    print("\nDetailed Results:")
    for result in results:
        print(f"\nPage {result['page_number']}:")
        print(f"  Classification: {result.get('classification', 'unknown')}")
        print(f"  Confidence: {result.get('confidence', 0):.2f}")
        print(f"  Reason: {result.get('reason', 'N/A')}")
        print(f"  Text length: {result.get('text_length', 0)}")
        
    return results

# Test with a sample PDF
if __name__ == "__main__":
    # You can test with your PDF files
    test_pdf_classification("docs_test/SOH - 131 VERDUN - Rapport revue HQE Phase PRO V2.pdf")