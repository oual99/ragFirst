"""Enhanced search functionality module with conversational features."""
import weaviate.classes.query as wq
from typing import List, Dict, Optional
import re
from datetime import datetime


class SearchEngine:
    def __init__(self, weaviate_client, embedding_generator):
        self.client = weaviate_client
        self.embedding_generator = embedding_generator
        self.search_history = []  # Track recent searches for context
    
    def search_multimodal(self, query: str, collection_name: str, limit: int = 3):
        """Perform vector search on the collection."""
        print(f"\n{'='*60}")
        print(f"🔍 SEARCH ENGINE - Starting search")
        print(f"Query: '{query}'")
        print(f"Collection: {collection_name}")
        print(f"Limit: {limit}")
        print(f"{'='*60}")
        
        # Generate embedding
        query_vector = self.embedding_generator.get_embedding(query)
        print(f"✅ Generated query embedding (dim: {len(query_vector)})")
        
        # Use the database's search method
        results = self.client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit
        )
        
        print(f"✅ Search completed - Found {len(results)} results")
        
        # Debug each result - results are already formatted as objects by the database layer
        for i, obj in enumerate(results):
            # Access properties directly as they're already objects
            distance = obj.metadata.distance if hasattr(obj.metadata, 'distance') else 0
            properties = obj.properties
            print(f"\n  Result {i+1}:")
            print(f"    Distance: {distance:.4f}")
            print(f"    Document: {properties.get('source_document', 'N/A')}")
            print(f"    Location: Page {properties.get('page_number')}, Para {properties.get('paragraph_number')}")
            print(f"    Text (50 chars): {properties.get('text', '')[:50]}...")
        
        print(f"{'='*60}\n")
        
        # Track search
        self.search_history.append({
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "result_count": len(results)
        })
        
        # Keep only last 10 searches
        if len(self.search_history) > 10:
            self.search_history = self.search_history[-10:]
        
        # Results are already in the correct format
        return results
    
    def _search_related_content(self, 
                               collection_name: str,
                               related_documents: set,
                               related_pages: Dict[str, List[int]],
                               limit: int):
        """Search for content from related documents or pages."""
        collection = self.client.collections.get(collection_name)
        related_results = []
        
        for doc in related_documents:
            if doc in related_pages:
                # Search for content from nearby pages
                for page in related_pages[doc]:
                    # Look for content from same page or adjacent pages
                    for page_offset in [0, -1, 1, -2, 2]:
                        target_page = page + page_offset
                        if target_page > 0:  # Valid page number
                            try:
                                results = collection.query.where(
                                    wq.Filter.by_property("source_document").equal(doc) &
                                    wq.Filter.by_property("page_number").equal(target_page)
                                ).with_limit(limit).do()
                                
                                related_results.extend(results.objects)
                                
                                if len(related_results) >= limit:
                                    return related_results[:limit]
                            except:
                                continue
        
        return related_results[:limit]
    
    def format_search_results(self, search_results) -> List[Dict]:
        """Format search results for display with enhanced metadata."""
        formatted_results = []
        
        for i, item in enumerate(search_results):
            result = {
                "source_document": item.properties['source_document'],
                "page_number": item.properties['page_number'],
                "paragraph_number": item.properties['paragraph_number'],
                "text": item.properties['text'],
                "distance": item.metadata.distance if hasattr(item.metadata, 'distance') else 0.0,
                "result_index": i + 1,
                "text_preview": self._create_text_preview(item.properties['text'], 150)
            }
            formatted_results.append(result)
        
        return formatted_results
    
    def _create_text_preview(self, text: str, max_length: int) -> str:
        """Create a preview of text for display."""
        if len(text) <= max_length:
            return text
        
        # Try to cut at a sentence boundary
        truncated = text[:max_length]
        last_period = truncated.rfind('.')
        if last_period > max_length * 0.8:  # If we have a period reasonably close to the end
            return truncated[:last_period + 1]
        
        # Otherwise, cut at word boundary
        last_space = truncated.rfind(' ')
        if last_space > 0:
            return truncated[:last_space] + "..."
        
        return truncated + "..."
    
    def extract_keywords_from_query(self, query: str) -> List[str]:
        """Extract important keywords from a query."""
        # Remove common words (simplified French stop words)
        stop_words = {
            'le', 'la', 'les', 'un', 'une', 'des', 'de', 'du', 'et', 'ou', 'à', 
            'dans', 'pour', 'sur', 'avec', 'par', 'est', 'sont', 'a', 'ai',
            'quel', 'quelle', 'quels', 'quelles', 'comment', 'où', 'quand',
            'je', 'tu', 'il', 'elle', 'nous', 'vous', 'ils', 'elles'
        }
        
        # Extract words
        words = re.findall(r'\b\w+\b', query.lower())
        
        # Filter out stop words and short words
        keywords = [word for word in words 
                   if word not in stop_words and len(word) > 2]
        
        # Prioritize technical terms
        technical_terms = {
            'ascenseur', 'bâtiment', 'étage', 'niveau', 'construction',
            'structure', 'dimension', 'matériau', 'béton', 'acier',
            'norme', 'sécurité', 'réglementation', 'certificat'
        }
        
        # Sort keywords by importance (technical terms first)
        keywords.sort(key=lambda x: 0 if x in technical_terms else 1)
        
        return keywords[:5]  # Return top 5 keywords
    
    def get_search_suggestions(self, partial_query: str, recent_searches: List[str]) -> List[str]:
        """Get search suggestions based on partial query and history."""
        suggestions = []
        partial_lower = partial_query.lower()
        
        # From recent searches
        for search in recent_searches:
            if partial_lower in search.lower() and search not in suggestions:
                suggestions.append(search)
        
        # Common BTP-related suggestions
        common_queries = [
            "nombre d'étages du bâtiment",
            "dimensions de l'ascenseur",
            "normes de sécurité",
            "matériaux de construction",
            "plan d'évacuation",
            "capacité maximale",
            "certification du bâtiment",
            "année de construction"
        ]
        
        for query in common_queries:
            if partial_lower in query.lower() and query not in suggestions:
                suggestions.append(query)
        
        return suggestions[:5]
    
    def analyze_search_pattern(self) -> Dict:
        """Analyze recent search patterns for insights."""
        if not self.search_history:
            return {"pattern": "no_history", "suggestions": []}
        
        # Extract topics from recent searches
        all_keywords = []
        for search in self.search_history[-5:]:
            keywords = self.extract_keywords_from_query(search['query'])
            all_keywords.extend(keywords)
        
        # Count keyword frequency
        keyword_freq = {}
        for kw in all_keywords:
            keyword_freq[kw] = keyword_freq.get(kw, 0) + 1
        
        # Identify pattern
        if not keyword_freq:
            pattern = "general_exploration"
        elif max(keyword_freq.values()) >= 3:
            # User is focused on specific topic
            pattern = "focused_research"
            main_topic = max(keyword_freq, key=keyword_freq.get)
        else:
            pattern = "broad_research"
        
        return {
            "pattern": pattern,
            "frequent_keywords": sorted(keyword_freq.items(), key=lambda x: x[1], reverse=True)[:3],
            "search_count": len(self.search_history)
        }