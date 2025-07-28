"""Enhanced search functionality module with conversational features."""
import weaviate.classes.query as wq
from typing import List, Dict, Optional
import re
from datetime import datetime
from typing import Any, List

class SearchEngine:
    def __init__(self, weaviate_client, embedding_generator):
        self.client = weaviate_client
        self.embedding_generator = embedding_generator
        self.search_history = []  # Track recent searches for context
    
    def rerank_results(self, query: str, search_results: List, top_k: int = 6) -> List:
        """
        Rerank search results using GPT-4 to select the most relevant ones.
        
        Args:
            query: The user's query
            search_results: List of search results to rerank
            top_k: Number of top results to return
            
        Returns:
            List of reranked results (top_k most relevant)
        """
        if not search_results or len(search_results) <= top_k:
            return search_results
        
        # Helper: grab a window around the first query-term hit
        def extract_snippet(text: str, terms: List[str],
                            window: int = 250, max_len: int = 700) -> str:
            lower = text.lower()
            for t in terms:
                pos = lower.find(t.lower())
                if pos != -1:
                    start = max(0, pos - window)
                    return text[start:start + max_len]
            return text[:max_len]
        
        # Build candidates with query-focused snippets
        query_terms = [t for t in query.split() if len(t) > 2]
        candidates = []
        for i, result in enumerate(search_results):
            # Check if result has 'properties' attribute (e.g., from an object) or is a dict directly
            props = result.properties if hasattr(result, 'properties') else result

            full_text = props.get('text', '')
            snippet = extract_snippet(full_text, query_terms)
            
            candidates.append({
                "index": i,
                "document": props.get('source_document', 'Unknown'),
                "page": props.get('page_number', 0),
                "text": snippet,
                "full_result": result  # Keep the full result for later
            })
        
        # Create reranking prompt
        rerank_prompt = f"""Tu es un expert en analyse de pertinence pour des documents BTP.

        Question de l'utilisateur: "{query}"

        Voici {len(candidates)} extraits de documents. Analyse leur pertinence par rapport à la question et sélectionne les {top_k} PLUS PERTINENTS.

        Critères de sélection:
        1. Pertinence directe avec la question
        2. Complétude de l'information
        3. Précision technique
        4. Contexte approprié
        5. Diversité : si plusieurs extraits du même document apportent la même information, privilégie ceux issus d’autres documents

        Documents candidats:
        """
        
        for i, candidate in enumerate(candidates):
            rerank_prompt += f"\n[Candidat {i+1}]\nDocument: {candidate['document']}, Page {candidate['page']}\nExtrait: {candidate['text']}\n"
        
        rerank_prompt += f"""
    Réponds UNIQUEMENT avec un JSON contenant:
    - "selected": liste des numéros des {top_k} candidats les plus pertinents (ex: [3, 1, 5, 9, 15])
    - "reasoning": brève explication de ton choix

    Format: {{"selected": [X, Y, Z], "reasoning": "..."}}"""
        
        # try:
        # Import OpenAI client
        from openai import OpenAI
        import config
        import json
        
        client = OpenAI(api_key=config.OPENAI_API_KEY)
        
        response = client.chat.completions.create(
            # model="gpt-4o-mini",  # Use mini for reranking to save costs
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Tu es un expert en analyse de pertinence. Réponds uniquement en JSON valide."},
                {"role": "user", "content": rerank_prompt}
            ],
            temperature=0.0,
            max_tokens=1000
        )
        
        # Parse response
        response_text = response.choices[0].message.content.strip()
        # Extract JSON even if there's extra text
        import re
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            rerank_data = json.loads(json_match.group())
            selected_indices = rerank_data.get("selected", [])
            
            # Get the selected results in order
            reranked_results = []
            for idx in selected_indices[:top_k]:
                if 1 <= idx <= len(candidates):
                    reranked_results.append(candidates[idx-1]["full_result"])
            
            # If we didn't get enough results, fill with top similarity results
            if len(reranked_results) < top_k:
                for i in range(min(top_k, len(search_results))):
                    if search_results[i] not in reranked_results:
                        reranked_results.append(search_results[i])
                    if len(reranked_results) >= top_k:
                        break
            
            print(f"\n🎯 Reranking: {len(search_results)} → {len(reranked_results)} results")
            print(f"Reasoning: {rerank_data.get('reasoning', 'N/A')}")
            print(f"Selected indices: {selected_indices[:top_k]}")
            
            
            
            # --- DEBUG: dump to files with numbering ---
            import os
            debug_dir = "debug_retrieval"
            os.makedirs(debug_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            uid = "default_user"
            #  reranking results
            reranking_path = os.path.join(debug_dir, f"{uid}_raranker_{ts}.txt")
            with open(reranking_path, "w", encoding="utf-8") as f:
                for i, chunk in enumerate(reranked_results, start=1):
                    props = chunk.properties if hasattr(chunk, 'properties') else chunk
                    full_text = props.get('text', '')
                    f.write(f"Chunk {i} (raranker):\n")
                    f.write(full_text + "\n\n")
            print(f"💾 Saved raranker chunks to {reranking_path}")
            
            return reranked_results
                
        # except Exception as e:
        #     print(f"❌ Reranking failed: {str(e)}")
        #     # Fallback to original results
        #     return search_results[:top_k]
        
        # Fallback: return top results by similarity
        
        
        
        return search_results[:top_k]
    



    # def search_multimodal(self, query: str, collection_name: str, limit: int = 3):
    #     """Perform vector search on the collection with deduplication."""
    #     print(f"\n{'='*60}")
    #     print(f"🔍 SEARCH ENGINE - Starting search")
    #     print(f"Query: '{query}'")
    #     print(f"Collection: {collection_name}")
    #     print(f"Requested limit: {limit}")
        
    #     # Over-fetch to account for duplicates
    #     search_limit = limit * 4
    #     print(f"Searching for: {search_limit} results (over-fetching for deduplication)")
    #     print(f"{'='*60}")
        
    #     # Generate embedding
    #     query_vector = self.embedding_generator.get_embedding(query)
    #     print(f"✅ Generated query embedding (dim: {len(query_vector)})")
        
    #     # Use the database's search method
    #     results = self.client.search(
    #         collection_name=collection_name,
    #         query_vector=query_vector,
    #         limit=search_limit
    #     )
    #     results.sort(key=lambda r: (r.properties.get('source_document', ''), r.properties.get('page_number', 0)))

        
    #     print(f"✅ Search completed - Found {len(results)} results")
        
    #     # Deduplicate based on first 1000 characters of text
    #     unique_results = []
    #     seen_texts = set()
    #     duplicates_removed = 0
        
    #     for obj in results:
    #         text_content = obj.properties.get('text', '')
    #         text_key = text_content[:1000]
            
    #         if text_key not in seen_texts:
    #             seen_texts.add(text_key)
    #             unique_results.append(obj)
    #         else:
    #             duplicates_removed += 1
        
    #     final_results = unique_results
        
    #     # print(f"🔄 Deduplication completed:")
    #     # print(f"   - Removed {duplicates_removed} duplicates")
    #     # print(f"   - {len(unique_results)} unique results found")
    #     # print(f"   - Returning {len(final_results)} results")
        
    #     for i, obj in enumerate(final_results):
    #         distance = obj.metadata.distance if hasattr(obj.metadata, 'distance') else 0
    #         props = obj.properties
    #         document = props.get('source_document', 'N/A')
    #         page = props.get('page_number', 'N/A')
    #         para = props.get('paragraph_number', 'N/A')
    #         text = props.get('text', '')[:300]

    #     #     print(f"\n  Result {i+1}:")
    #     #     print(f"    Distance: {distance:.4f}")
    #         print(f"    Document: {document}")
    #         print(f"    Location: Page {page}, Para {para}")
    #     #     print(f"    Text (300 chars): {text}...")
        
    #     # print(f"{'='*60}\n")
        
    #     # Track search
    #     self.search_history.append({
    #         "query": query,
    #         "timestamp": datetime.now().isoformat(),
    #         "raw_result_count": len(results),
    #         "duplicates_removed": duplicates_removed,
    #         "final_result_count": len(final_results)
    #     })
        
    #     # Keep only last 10 searches
    #     if len(self.search_history) > 10:
    #         self.search_history = self.search_history[-10:]
        
    #     # Results are already in the correct format
    #     return final_results
        
    
    def _rrf_merge(self,
               emb_results: List[Any],
               bm25_results: List[Any],
               limit: int,
               k: int = 50,
               w_emb: float = 1.2,
               w_bm25: float = 0.8):
        """
        Reciprocal Rank Fusion of two ranked lists with global de‑duplication.
        `limit` = number of items to return.
        `k`     = RRF dampening constant (50 is conventional).
        `w_*`   = optional per‑retriever weight.
        """
        def _extract_key_and_text(item: Any) -> tuple[str, str]:
            """
            Works for:
            • dict‑like items   → item["id"], item["text"]
            • ResultObject      → item.id or .uuid, item.text or item.properties["text"]
            Falls back to text fingerprint.
            """
            def _fingerprint(text: str) -> str:
                """Whitespace‑&‑case‑normalised MD5 of a chunk for fast de‑dup."""
                import hashlib
                import re
                norm = re.sub(r"\W+", "", text.lower())
                return hashlib.md5(norm.encode()).hexdigest()
            # --- 1. Get text --------------------------------------------------------
            text = None
            if isinstance(item, dict):
                text = item.get("text") or item.get("chunk")
                key  = item.get("id")
            else:                                   # assume class instance
                # Try common attribute names
                text = getattr(item, "text", None)
                if text is None and hasattr(item, "properties"):
                    text = item.properties.get("text")

                key = getattr(item, "id", None) or getattr(item, "uuid", None)
            if text is None:
                text = ""          # should not happen, but keep code safe

            # --- 2. Derive key ------------------------------------------------------
            if key is None:
                key = _fingerprint(text)            # dedup key of last resort
            return key, text
        from collections import defaultdict
        scores   = defaultdict(float)
        payload  = {}

        for rank, item in enumerate(emb_results, start=1):
            key, _ = _extract_key_and_text(item)
            payload.setdefault(key, item)                     # keep first seen
            scores[key] += w_emb / (k + rank)

        for rank, item in enumerate(bm25_results, start=1):
            key, _ = _extract_key_and_text(item)
            payload.setdefault(key, item)
            scores[key] += w_bm25 / (k + rank)

        fused = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return [payload[key] for key, _ in fused[:limit]]
    def _merge_embeddings_bm25_rrf(self, emb_results, bm25_results, limit):
        return self._rrf_merge(emb_results, bm25_results, limit)

    # def _merge_half_and_half(self, emb_results, bm25_results, limit):
    #     """
    #     Take half of `limit` from embeddings, half from BM25, dedupe,
    #     then fill any remaining slots from the union (preserving score order).
    #     """
    #     # 1. Determine how many slots for each
    #     half = limit // 2
    #     emb_quota  = emb_results[:half]
    #     bm25_quota = bm25_results[: limit - half]

    #     # 2. Start with embeddings quota, then BM25 quota, but drop duplicates
    #     final = []
    #     for item in emb_quota + bm25_quota:
    #         if item not in final:
    #             final.append(item)

    #     # 3. If we still need more to reach `limit`, pull from the rest, preserving original order
    #     if len(final) < limit:
    #         pool = emb_results + bm25_results
    #         for item in pool:
    #             if len(final) >= limit:
    #                 break
    #             if item not in final:
    #                 final.append(item)

    #     return final

    def search_multimodal(self,
                        query: str,
                        collection_name: str,
                        limit: int = 3,
                        mode: str = "Hybrid",
                        user_id: str = None):
        import numpy as np
        from datetime import datetime
        from src.bm25_utils import load_bm25
        """
        Unified retrieval: embeddings-only, BM25-only, or hybrid.

        Args:
            query: user question
            collection_name: Qdrant collection name
            limit: how many final results to return
            mode: "Embeddings only", "BM25 only", or "Hybrid"
            user_id: identifier to load the correct BM25 index
        Returns:
            List of chunk objects (either Qdrant hits or BM25 chunk dicts)
        """
        if mode is None:
            mode = "Hybrid"
        mode_key = mode.lower().replace(" ", "_")  # embeddings_only, bm25_only, hybrid

        emb_results = []
        bm25_results = []

        # ─── Embeddings search ────────────────────────────────────────────────
        if mode_key in ("embeddings_only", "hybrid"):
            query_vector = self.embedding_generator.get_embedding(query)
            raw = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit * 4
            )
            # dedupe
            seen = set()
            unique = []
            for obj in raw:
                key = obj.properties.get("text", "")[:1000]
                if key not in seen:
                    seen.add(key)
                    unique.append(obj)
            emb_results = unique[:limit]

        # ─── BM25 search ──────────────────────────────────────────────────────
        if mode_key in ("bm25_only", "hybrid") and user_id:
            bm25, docs = load_bm25(user_id)
            if bm25 and docs:
                tokens = query.lower().split()
                scores = bm25.get_scores(tokens)
                top_idx = np.argsort(scores)[-limit:][::-1]
                
                seen_chunks = set()
                bm25_results = []
                
                for i in top_idx:
                    chunk = docs[i]["text"]
                    chunk_start = chunk[:500]
                    if chunk_start not in seen_chunks:
                        seen_chunks.add(chunk_start)
                        bm25_results.append(docs[i])

        # ─── Merge or pick one set ────────────────────────────────────────────
        if mode_key == "embeddings_only":
            final = emb_results
            print(f"🔍 Embeddings search returned {len(final)} results")
        elif mode_key == "bm25_only":
            final = bm25_results
            print(f"🔍 BM25 search returned {len(final)} results")
        else:  # hybrid
            final = self._merge_embeddings_bm25_rrf(emb_results, bm25_results,limit = 40)
            print(f"🔍 Hybrid search returned {len(final)} results (BM25: {len(bm25_results)}, Embeddings: {len(emb_results)})")
            
            # --- DEBUG: dump to files with numbering ---
            import os
            debug_dir = "debug_retrieval"
            os.makedirs(debug_dir, exist_ok=True)

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            uid = user_id or "anon"

            def get_text(item):
                if hasattr(item, "properties"):
                    return item.properties.get("text", "")
                elif isinstance(item, dict):
                    return item.get("text", "")
                else:
                    return str(item)

            # 1) BM25 results
            bm25_path = os.path.join(debug_dir, f"{uid}_bm25_{ts}.txt")
            with open(bm25_path, "w", encoding="utf-8") as f:
                for i, chunk in enumerate(bm25_results, start=1):
                    f.write(f"Chunk {i} (BM25):\n")
                    f.write(get_text(chunk).strip() + "\n\n")
            print(f"💾 Saved BM25 chunks to {bm25_path}")

            # 2) Embedding results
            emb_path = os.path.join(debug_dir, f"{uid}_emb_{ts}.txt")
            with open(emb_path, "w", encoding="utf-8") as f:
                for i, chunk in enumerate(emb_results, start=1):
                    f.write(f"Chunk {i} (Embedding):\n")
                    f.write(get_text(chunk).strip() + "\n\n")
            print(f"💾 Saved embedding chunks to {emb_path}")

            # 3) Final merged results
            hybrid_path = os.path.join(debug_dir, f"{uid}_hybrid_{ts}.txt")
            with open(hybrid_path, "w", encoding="utf-8") as f:
                for i, chunk in enumerate(final, start=1):
                    f.write(f"Chunk {i} (Final):\n")
                    f.write(get_text(chunk).strip() + "\n\n")
            print(f"💾 Saved merged chunks to {hybrid_path}")

        # ─── Logging ─────────────────────────────────────────────────────────
        self.search_history.append({
            "query": query,
            "mode": mode,
            "user_id": user_id,
            "timestamp": datetime.now().isoformat(),
            "results_returned": len(final)
        })
        if len(self.search_history) > 50:
            self.search_history = self.search_history[-50:]

        return final



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
    
    def format_search_results(self, results):
        """
        Normalize Qdrant hits and BM25 chunk dicts into a common list of metadata dicts.
        """
        formatted = []
        for item in results:
            # 1. Distinguish Qdrant result vs BM25 chunk dict
            if hasattr(item, "properties"):
                # Qdrant hit
                props = item.properties
                score = getattr(item.metadata, "distance", None)
            elif isinstance(item, dict):
                # BM25 chunk dict
                props = item
                score = None
            else:
                # Fallback: treat as text-only
                props = {"text": str(item)}
                score = None

            # 2. Build your formatted entry
            formatted.append({
                "source_document": props.get("source_document"),
                "page_number":    props.get("page_number"),
                "paragraph_number": props.get("paragraph_number"),
                "text":           props.get("text"),
                "score":          score,
            })

        return formatted
    
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