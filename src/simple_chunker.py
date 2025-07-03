# src/simple_chunker.py
import tiktoken
from typing import List, Dict

class SimpleChunker:
    def __init__(self, chunk_size: int, overlap_size: int):
        """
        Initialize the chunker with token-based sizes.
        
        Args:
            chunk_size: Target size in tokens (default 500)
            overlap_size: Overlap between chunks in tokens (default 100)
        """
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        # Use OpenAI's tokenizer (same as text-embedding-3-large uses)
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def chunk_text(self, text: str, metadata: Dict = None) -> List[Dict]:
        """
        Split text into chunks based on token count.
        
        Args:
            text: Text to chunk
            metadata: Base metadata to include with each chunk
            
        Returns:
            List of chunks with metadata
        """
        if not text or not text.strip():
            return []
        
        # Tokenize the entire text
        tokens = self.tokenizer.encode(text)
        
        chunks = []
        chunk_index = 0
        start_idx = 0
        
        while start_idx < len(tokens):
            # Get chunk tokens
            end_idx = min(start_idx + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start_idx:end_idx]
            
            # Decode back to text
            chunk_text = self.tokenizer.decode(chunk_tokens)
            
            # Create chunk with metadata
            chunk_data = {
                "text": chunk_text,
                "chunk_index": chunk_index,
                "start_token": start_idx,
                "end_token": end_idx,
                "token_count": len(chunk_tokens)
            }
            
            # Add any provided metadata
            if metadata:
                chunk_data.update(metadata)
            
            chunks.append(chunk_data)
            
            # Move to next chunk with overlap
            start_idx = end_idx - self.overlap_size
            chunk_index += 1
            
            # Prevent infinite loop if overlap >= chunk_size
            if start_idx <= chunk_index * self.overlap_size:
                start_idx = end_idx
        
        return chunks
    
    def process_document_results(self, unified_results: Dict) -> List[Dict]:
        """
        Process results from UnifiedDocumentProcessor into chunks with cross-page continuity.
        
        Args:
            unified_results: Results from UnifiedDocumentProcessor
            
        Returns:
            List of chunks ready for indexing
        """
        # APPROACH 1: Concatenate all pages first, then chunk
        return self._process_with_full_concatenation(unified_results)
        
        # Alternative approaches (uncomment to use):
        # return self._process_with_sliding_window(unified_results)
        # return self._process_with_smart_boundaries(unified_results)
    
    def _process_with_full_concatenation(self, unified_results: Dict) -> List[Dict]:
        """
        Concatenate all page content, then chunk the entire document.
        Best for: Documents where page breaks are arbitrary (like PDFs).
        """
        # Collect all successful pages
        pages_content = []
        page_mappings = []
        
        for page in unified_results["pages"]:
            if page.get("status") != "success" or not page.get("content"):
                continue
            
            page_content = page["content"].strip()
            if page_content:
                pages_content.append({
                    'content': page_content,
                    'page_number': page["page_number"],
                    'page_type': page.get("type", "unknown"),
                    'processing_method': page.get("processing_method", "unknown")
                })
        
        if not pages_content:
            return []
        
        # Join all content with space separation
        full_text = ' '.join([p['content'] for p in pages_content])
        
        # Create character position mappings for pages
        char_pos = 0
        for page_info in pages_content:
            start_pos = char_pos
            char_pos += len(page_info['content']) + 1  # +1 for space
            end_pos = char_pos - 1
            
            page_mappings.append({
                'page_number': page_info['page_number'],
                'page_type': page_info['page_type'],
                'processing_method': page_info['processing_method'],
                'start_char': start_pos,
                'end_char': end_pos
            })
        
        # Create base metadata with default values
        base_metadata = {
            'source_document': unified_results["document_name"]
        }
        
        # Chunk the entire document
        chunks = self.chunk_text(full_text, base_metadata)
        
        # More accurate page mapping using token positions
        for chunk in chunks:
            # Convert token positions to approximate character positions
            start_token = chunk.get('start_token', 0)
            end_token = chunk.get('end_token', 0)
            
            # Get the actual text slice to find character positions
            chunk_text = chunk['text']
            chunk_start_char = full_text.find(chunk_text[:50])  # Find chunk start in full text
            if chunk_start_char == -1:
                chunk_start_char = start_token * 3  # Fallback: rough estimate
            chunk_end_char = chunk_start_char + len(chunk_text)
            
            # Find which pages this chunk spans
            chunk_pages = []
            for page_info in page_mappings:
                if (chunk_start_char < page_info['end_char'] and 
                    chunk_end_char > page_info['start_char']):
                    chunk_pages.append(page_info)
            
            if chunk_pages:
                # Primary page (where most of the chunk content starts)
                primary_page = chunk_pages[0]
                chunk['page_number'] = primary_page['page_number']
                chunk['page_type'] = primary_page['page_type']
                chunk['processing_method'] = primary_page['processing_method']
                
                # Additional metadata for cross-page chunks
                if len(chunk_pages) > 1:
                    chunk['spans_pages'] = [p['page_number'] for p in chunk_pages]
                    chunk['is_cross_page'] = True
                else:
                    chunk['is_cross_page'] = False
            else:
                # Fallback: ensure we always have required fields
                chunk['page_number'] = pages_content[0]['page_number']
                chunk['page_type'] = pages_content[0]['page_type']
                chunk['processing_method'] = pages_content[0]['processing_method']
                chunk['is_cross_page'] = False
                print(f"Warning: Chunk {chunk['chunk_index']} couldn't be mapped to any page")
            
            # Keep paragraph numbering for compatibility
            chunk['paragraph_number'] = chunk['chunk_index'] + 1
        
        # Save chunks to file
        self._save_chunks_to_file(chunks)
        
        return chunks
    
    def _process_with_sliding_window(self, unified_results: Dict) -> List[Dict]:
        """
        Process pages with a sliding window that includes content from adjacent pages.
        Best for: When you want to maintain some page-level organization.
        """
        all_chunks = []
        pages = [p for p in unified_results["pages"] 
                if p.get("status") == "success" and p.get("content")]
        
        for i, page in enumerate(pages):
            # Get context from previous and next pages
            context_before = ""
            context_after = ""
            
            # Add content from previous page (last N tokens)
            if i > 0:
                prev_content = pages[i-1]["content"]
                prev_tokens = self.tokenizer.encode(prev_content)
                if len(prev_tokens) > self.overlap_size:
                    context_tokens = prev_tokens[-self.overlap_size:]
                    context_before = self.tokenizer.decode(context_tokens)
                else:
                    context_before = prev_content
            
            # Add content from next page (first N tokens)
            if i < len(pages) - 1:
                next_content = pages[i+1]["content"]
                next_tokens = self.tokenizer.encode(next_content)
                if len(next_tokens) > self.overlap_size:
                    context_tokens = next_tokens[:self.overlap_size]
                    context_after = self.tokenizer.decode(context_tokens)
                else:
                    context_after = next_content
            
            # Combine content with context
            full_content = f"{context_before} {page['content']} {context_after}".strip()
            
            # Create metadata
            metadata = {
                'source_document': unified_results["document_name"],
                'page_number': page["page_number"],
                'page_type': page.get("type", "unknown"),
                'processing_method': page.get("processing_method", "unknown"),
                'has_prev_context': bool(context_before),
                'has_next_context': bool(context_after)
            }
            
            # Chunk this enhanced content
            page_chunks = self.chunk_text(full_content, metadata)
            
            # Add paragraph numbering
            for j, chunk in enumerate(page_chunks):
                chunk['paragraph_number'] = j + 1
            
            all_chunks.extend(page_chunks)
        
        # Renumber chunks globally
        for i, chunk in enumerate(all_chunks):
            chunk['chunk_index'] = i
        
        # Save chunks to file
        self._save_chunks_to_file(all_chunks)
        
        return all_chunks
    
    def _process_with_smart_boundaries(self, unified_results: Dict) -> List[Dict]:
        """
        Concatenate pages but try to respect natural boundaries (sentences, paragraphs).
        Best for: Documents with clear structural elements.
        """
        import re
        
        # Collect all content with boundary markers
        all_content = []
        page_mappings = []
        
        for page in unified_results["pages"]:
            if page.get("status") != "success" or not page.get("content"):
                continue
            
            page_content = page["content"].strip()
            if page_content:
                # Add page boundary marker
                content_with_marker = f"\n[PAGE_BREAK:{page['page_number']}]\n{page_content}"
                all_content.append(content_with_marker)
                
                page_mappings.append({
                    'page_number': page["page_number"],
                    'page_type': page.get("type", "unknown"),
                    'processing_method': page.get("processing_method", "unknown")
                })
        
        if not all_content:
            return []
        
        full_text = '\n'.join(all_content)
        
        # Custom chunking that respects sentence boundaries
        chunks = self._chunk_with_sentence_awareness(
            full_text, 
            unified_results["document_name"],
            page_mappings
        )
        
        # Save chunks to file
        self._save_chunks_to_file(chunks)
        
        return chunks
    
    def _chunk_with_sentence_awareness(self, text: str, document_name: str, page_mappings: List[Dict]) -> List[Dict]:
        """
        Chunk text while trying to keep sentences intact.
        """
        import re
        
        # Split into sentences (simple approach)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk_tokens = []
        current_chunk_text = ""
        chunk_index = 0
        
        for sentence in sentences:
            sentence_tokens = self.tokenizer.encode(sentence)
            
            # Check if adding this sentence would exceed chunk size
            if (len(current_chunk_tokens) + len(sentence_tokens) > self.chunk_size 
                and current_chunk_tokens):  # Don't split if current chunk is empty
                
                # Finalize current chunk
                chunk_data = {
                    "text": current_chunk_text.strip(),
                    "chunk_index": chunk_index,
                    "token_count": len(current_chunk_tokens),
                    "source_document": document_name,
                }
                
                # Extract page info from chunk text
                page_matches = re.findall(r'\[PAGE_BREAK:(\d+)\]', current_chunk_text)
                if page_matches:
                    primary_page = int(page_matches[0])
                    chunk_data['page_number'] = primary_page
                    chunk_data['spans_pages'] = [int(p) for p in set(page_matches)]
                    chunk_data['is_cross_page'] = len(set(page_matches)) > 1
                    
                    # Find page metadata
                    for page_info in page_mappings:
                        if page_info['page_number'] == primary_page:
                            chunk_data.update({
                                'page_type': page_info['page_type'],
                                'processing_method': page_info['processing_method']
                            })
                            break
                
                # Clean up page markers from final text
                chunk_data['text'] = re.sub(r'\[PAGE_BREAK:\d+\]\s*', '', chunk_data['text'])
                chunk_data['paragraph_number'] = chunk_index + 1
                
                chunks.append(chunk_data)
                
                # Start new chunk with overlap
                if self.overlap_size > 0 and len(current_chunk_tokens) > self.overlap_size:
                    overlap_tokens = current_chunk_tokens[-self.overlap_size:]
                    overlap_text = self.tokenizer.decode(overlap_tokens)
                    current_chunk_tokens = overlap_tokens
                    current_chunk_text = overlap_text + " " + sentence
                else:
                    current_chunk_tokens = sentence_tokens
                    current_chunk_text = sentence
                
                chunk_index += 1
            else:
                # Add sentence to current chunk
                if current_chunk_text:
                    current_chunk_text += " " + sentence
                else:
                    current_chunk_text = sentence
                current_chunk_tokens.extend(sentence_tokens)
        
        # Handle final chunk
        if current_chunk_tokens:
            chunk_data = {
                "text": re.sub(r'\[PAGE_BREAK:\d+\]\s*', '', current_chunk_text.strip()),
                "chunk_index": chunk_index,
                "token_count": len(current_chunk_tokens),
                "source_document": document_name,
                "paragraph_number": chunk_index + 1
            }
            
            # Extract page info
            page_matches = re.findall(r'\[PAGE_BREAK:(\d+)\]', current_chunk_text)
            if page_matches:
                primary_page = int(page_matches[0])
                chunk_data['page_number'] = primary_page
                chunk_data['spans_pages'] = [int(p) for p in set(page_matches)]
                chunk_data['is_cross_page'] = len(set(page_matches)) > 1
                
                for page_info in page_mappings:
                    if page_info['page_number'] == primary_page:
                        chunk_data.update({
                            'page_type': page_info['page_type'],
                            'processing_method': page_info['processing_method']
                        })
                        break
            
            chunks.append(chunk_data)
        
        return chunks
    
    def _save_chunks_to_file(self, all_chunks: List[Dict]):
        """Save chunks to file for debugging."""
        try:
            filename = "./chunks.txt"
            with open(filename, "w", encoding="utf-8") as f:
                for c in all_chunks:
                    # Write header with more info
                    header = f"=== Chunk {c['chunk_index']} ==="
                    if c.get('is_cross_page'):
                        header += f" (Pages: {c.get('spans_pages', [])})"
                    else:
                        header += f" (Page: {c.get('page_number', 'Unknown')})"
                    header += f" (Tokens: {c.get('token_count', 0)}) ==="
                    
                    f.write(header + "\n")
                    f.write(c["text"].strip() + "\n\n")
            print(f"Saved {len(all_chunks)} chunks to '{filename}'")
        except IOError as e:
            print(f"Failed to write chunks to file: {e}")