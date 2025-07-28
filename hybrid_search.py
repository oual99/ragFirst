import os
import pickle
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np
from rank_bm25 import BM25Okapi
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Unified search result structure"""
    document: str
    page_number: int
    paragraph_number: int
    text: str
    score: float
    search_type: str  # 'bm25', 'vector', or 'hybrid'
    original_rank: int
    qdrant_id: str = None
    metadata: Dict = None

class BM25Retriever:
    """BM25 retrieval component for hybrid search"""
    
    def __init__(self, cache_path: str = "bm25_cache"):
        self.cache_path = cache_path
        self.bm25_index = None
        self.documents = []
        self.doc_metadata = []
        self.last_update = None
        self.collection_size = 0
        
    def _get_cache_file(self, collection_name: str) -> str:
        """Get cache file path for collection"""
        os.makedirs(self.cache_path, exist_ok=True)
        return os.path.join(self.cache_path, f"{collection_name}_bm25.pkl")
    
    def _needs_rebuild(self, collection_name: str, current_size: int) -> bool:
        """Check if BM25 index needs rebuilding"""
        cache_file = self._get_cache_file(collection_name)
        
        if not os.path.exists(cache_file):
            return True
            
        if self.bm25_index is None:
            return True
            
        if current_size != self.collection_size:
            return True
            
        return False
    
    def _load_cache(self, collection_name: str) -> bool:
        """Load BM25 index from cache"""
        cache_file = self._get_cache_file(collection_name)
        
        try:
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                self.bm25_index = cache_data['bm25_index']
                self.documents = cache_data['documents']
                self.doc_metadata = cache_data['doc_metadata']
                self.collection_size = cache_data['collection_size']
                self.last_update = cache_data['last_update']
                logger.info(f"✅ Loaded BM25 cache for {collection_name} ({len(self.documents)} docs)")
                return True
        except Exception as e:
            logger.warning(f"⚠️ Could not load BM25 cache: {e}")
            return False
    
    def _save_cache(self, collection_name: str):
        """Save BM25 index to cache"""
        cache_file = self._get_cache_file(collection_name)
        
        try:
            cache_data = {
                'bm25_index': self.bm25_index,
                'documents': self.documents,
                'doc_metadata': self.doc_metadata,
                'collection_size': self.collection_size,
                'last_update': datetime.now().isoformat()
            }
            
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
                logger.info(f"💾 Saved BM25 cache for {collection_name}")
        except Exception as e:
            logger.error(f"❌ Could not save BM25 cache: {e}")
    
    def build_index(self, qdrant_client, collection_name: str, embedding_generator):
        """Build BM25 index from Qdrant collection"""
        logger.info(f"🔨 Building BM25 index for collection: {collection_name}")
        
        # Get collection info
        collection_info = qdrant_client.get_collection(collection_name)
        current_size = collection_info.points_count
        
        # Check if rebuild is needed
        if not self._needs_rebuild(collection_name, current_size):
            logger.info("📋 BM25 index is up to date")
            return
        
        # Try to load from cache first
        if self._load_cache(collection_name):
            if current_size == self.collection_size:
                return
        
        logger.info(f"🔄 Rebuilding BM25 index for {current_size} documents...")
        
        # Fetch all documents from Qdrant
        try:
            # Scroll through all points in the collection
            points = []
            offset = None
            
            while True:
                result = qdrant_client.scroll(
                    collection_name=collection_name,
                    limit=1000,  # Fetch in batches
                    offset=offset,
                    with_payload=True,
                    with_vectors=False  # We don't need vectors for BM25
                )
                
                if not result[0]:  # No more points
                    break
                    
                points.extend(result[0])
                offset = result[1]  # Next offset
                
                if len(points) % 5000 == 0:
                    logger.info(f"  📥 Fetched {len(points)} documents...")
            
            logger.info(f"📦 Retrieved {len(points)} documents from Qdrant")
            
            # Prepare documents for BM25
            self.documents = []
            self.doc_metadata = []
            
            for point in points:
                text = point.payload.get('text', '')
                if text.strip():  # Only add non-empty texts
                    # Simple tokenization for BM25
                    tokens = text.lower().split()
                    self.documents.append(tokens)
                    
                    # Store metadata
                    self.doc_metadata.append({
                        'id': point.id,
                        'text': text,
                        'source_document': point.payload.get('source_document', ''),
                        'page_number': point.payload.get('page_number', 0),
                        'paragraph_number': point.payload.get('paragraph_number', 0),
                        'original_payload': point.payload
                    })
            
            # Build BM25 index
            if self.documents:
                self.bm25_index = BM25Okapi(self.documents)
                self.collection_size = len(self.documents)
                logger.info(f"✅ Built BM25 index with {len(self.documents)} documents")
                
                # Save to cache
                self._save_cache(collection_name)
            else:
                logger.warning("⚠️ No documents found for BM25 indexing")
                
        except Exception as e:
            logger.error(f"❌ Error building BM25 index: {e}")
            raise
    
    def search(self, query: str, limit: int = 50) -> List[SearchResult]:
        """Search using BM25"""
        if not self.bm25_index or not self.documents:
            logger.warning("⚠️ BM25 index not built")
            return []
        
        # Tokenize query
        query_tokens = query.lower().split()
        
        # Get BM25 scores
        scores = self.bm25_index.get_scores(query_tokens)
        
        # Get top results
        top_indices = np.argsort(scores)[::-1][:limit]
        
        results = []
        for rank, idx in enumerate(top_indices):
            if scores[idx] > 0:  # Only include results with positive scores
                metadata = self.doc_metadata[idx]
                
                result = SearchResult(
                    document=metadata['source_document'],
                    page_number=metadata['page_number'],
                    paragraph_number=metadata['paragraph_number'],
                    text=metadata['text'],
                    score=float(scores[idx]),
                    search_type='bm25',
                    original_rank=rank + 1,
                    qdrant_id=metadata['id'],
                    metadata=metadata['original_payload']
                )
                results.append(result)
        
        logger.info(f"🔍 BM25 search returned {len(results)} results")
        return results

class HybridSearcher:
    """Hybrid search combining BM25 and vector search"""
    
    def __init__(self, qdrant_client, embedding_generator, cache_path: str = "bm25_cache"):
        self.qdrant_client = qdrant_client
        self.embedding_generator = embedding_generator
        self.bm25_retriever = BM25Retriever(cache_path)
        self.search_history = []
    
    def _vector_results_to_search_results(self, vector_results) -> List[SearchResult]:
        """Convert vector search results to SearchResult objects"""
        results = []
        
        for rank, obj in enumerate(vector_results):
            # Handle different result formats
            if hasattr(obj, 'payload'):
                payload = obj.payload
                point_id = obj.id
                # Distance is typically in metadata
                score = 1.0 - (obj.score if hasattr(obj, 'score') else 0.0)  # Convert distance to similarity
            else:
                # Handle properties format
                payload = obj.properties if hasattr(obj, 'properties') else {}
                point_id = getattr(obj, 'id', str(rank))
                score = 1.0 - (obj.metadata.distance if hasattr(obj.metadata, 'distance') else 0.0)
            
            result = SearchResult(
                document=payload.get('source_document', 'N/A'),
                page_number=payload.get('page_number', 0),
                paragraph_number=payload.get('paragraph_number', 0),
                text=payload.get('text', ''),
                score=score,
                search_type='vector',
                original_rank=rank + 1,
                qdrant_id=point_id,
                metadata=payload
            )
            results.append(result)
        
        return results
    
    def _reciprocal_rank_fusion(self, 
                              bm25_results: List[SearchResult], 
                              vector_results: List[SearchResult],
                              k: int = 60,
                              bm25_weight: float = 0.5,
                              vector_weight: float = 0.5) -> List[SearchResult]:
        """Apply Reciprocal Rank Fusion to combine results"""
        
        # Create a mapping of document content to results
        result_map = {}
        
        # Process BM25 results
        for rank, result in enumerate(bm25_results):
            key = f"{result.document}_{result.page_number}_{result.paragraph_number}"
            
            if key not in result_map:
                result_map[key] = {
                    'result': result,
                    'bm25_rank': rank + 1,
                    'vector_rank': None,
                    'bm25_score': result.score,
                    'vector_score': 0.0
                }
            else:
                result_map[key]['bm25_rank'] = rank + 1
                result_map[key]['bm25_score'] = result.score
        
        # Process vector results
        for rank, result in enumerate(vector_results):
            key = f"{result.document}_{result.page_number}_{result.paragraph_number}"
            
            if key not in result_map:
                result_map[key] = {
                    'result': result,
                    'bm25_rank': None,
                    'vector_rank': rank + 1,
                    'bm25_score': 0.0,
                    'vector_score': result.score
                }
            else:
                result_map[key]['vector_rank'] = rank + 1
                result_map[key]['vector_score'] = result.score
        
        # Calculate RRF scores
        for key, data in result_map.items():
            rrf_score = 0.0
            
            # BM25 contribution
            if data['bm25_rank'] is not None:
                rrf_score += bm25_weight * (1.0 / (k + data['bm25_rank']))
            
            # Vector contribution
            if data['vector_rank'] is not None:
                rrf_score += vector_weight * (1.0 / (k + data['vector_rank']))
            
            # Update result with hybrid score
            data['result'].score = rrf_score
            data['result'].search_type = 'hybrid'
        
        # Sort by RRF score and return top results
        sorted_results = sorted(result_map.values(), key=lambda x: x['result'].score, reverse=True)
        
        return [item['result'] for item in sorted_results]
    
    def search_hybrid(self, query: str, collection_name: str, limit: int = 30) -> List[SearchResult]:
        """
        Perform hybrid search combining BM25 and vector search
        
        Args:
            query: Search query
            collection_name: Qdrant collection name
            limit: Number of final results to return (default: 30)
            
        Returns:
            List of SearchResult objects ranked by hybrid score
        """
        print(f"\n{'='*60}")
        print(f"🔍 HYBRID SEARCH ENGINE - Starting search")
        print(f"Query: '{query}'")
        print(f"Collection: {collection_name}")
        print(f"Final limit: {limit}")
        print(f"{'='*60}")
        
        start_time = datetime.now()
        
        try:
            # Step 1: Build/update BM25 index
            print("🔨 Building/updating BM25 index...")
            self.bm25_retriever.build_index(self.qdrant_client, collection_name, self.embedding_generator)
            
            # Step 2: BM25 search
            print("📖 Performing BM25 search...")
            bm25_results = self.bm25_retriever.search(query, limit=50)
            print(f"✅ BM25 returned {len(bm25_results)} results")
            
            # Step 3: Vector search (reuse existing search_multimodal)
            print("🧮 Performing vector search...")
            vector_raw_results = self.qdrant_client.search_multimodal(query, collection_name, limit=50)
            vector_results = self._vector_results_to_search_results(vector_raw_results)
            print(f"✅ Vector search returned {len(vector_results)} results")
            
            # Step 4: Hybrid fusion
            print("🔄 Applying Reciprocal Rank Fusion...")
            hybrid_results = self._reciprocal_rank_fusion(
                bm25_results, 
                vector_results,
                bm25_weight=0.5,
                vector_weight=0.5
            )
            
            # Step 5: Apply final limit
            final_results = hybrid_results[:limit]
            
            # Timing
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            print(f"✅ Hybrid search completed in {duration:.2f}s")
            print(f"📊 Final results: {len(final_results)}")
            print(f"   - BM25 contributed: {len([r for r in final_results if r.search_type == 'bm25' or any(rb.qdrant_id == r.qdrant_id for rb in bm25_results)])}")
            print(f"   - Vector contributed: {len([r for r in final_results if r.search_type == 'vector' or any(rv.qdrant_id == r.qdrant_id for rv in vector_results)])}")
            print(f"   - Hybrid fusion: {len([r for r in final_results if r.search_type == 'hybrid'])}")
            
            # Show top results
            print("\n🏆 Top 5 results:")
            for i, result in enumerate(final_results[:5]):
                print(f"  {i+1}. {result.document} (Page {result.page_number}) - Score: {result.score:.4f}")
                print(f"     Type: {result.search_type} | Text: {result.text[:100]}...")
            
            print(f"{'='*60}\n")
            
            # Track search history
            self.search_history.append({
                "query": query,
                "timestamp": start_time.isoformat(),
                "bm25_results": len(bm25_results),
                "vector_results": len(vector_results),
                "final_results": len(final_results),
                "duration": duration
            })
            
            # Keep only last 10 searches
            if len(self.search_history) > 10:
                self.search_history = self.search_history[-10:]
            
            return final_results
            
        except Exception as e:
            logger.error(f"❌ Error in hybrid search: {e}")
            print(f"❌ Hybrid search failed: {e}")
            print("🔄 Falling back to vector search...")
            
            # Fallback to vector search
            vector_raw_results = self.qdrant_client.search_multimodal(query, collection_name, limit=limit)
            return self._vector_results_to_search_results(vector_raw_results)

# Integration helper functions
def add_hybrid_search_to_existing_class(existing_db, embedding_generator):
    """
    Add a hybrid-search method to the existing DB instance,
    letting HybridSearcher talk directly to the low‑level client.
    """
    def search_hybrid(self, query: str, collection_name: str, limit: int = 30):
        # Lazy‑init the HybridSearcher against the underlying client
        if not hasattr(self, '_hybrid_searcher'):
            self._hybrid_searcher = HybridSearcher(
                qdrant_client=self.client,          # low‑level Qdrant client
                embedding_generator=embedding_generator
            )
        return self._hybrid_searcher.search_hybrid(query, collection_name, limit)

    # Bind the new method on just this instance
    existing_db.search_hybrid = search_hybrid.__get__(existing_db)
    return existing_db
