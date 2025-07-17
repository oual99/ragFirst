"""Query expansion module for RAG system with BTP terminology."""
import pandas as pd
import re
import json
import logging
from typing import List, Dict, Set, Optional, Tuple
from pathlib import Path
import openai
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class QueryExpansionEngine:
    """Engine for expanding queries using BTP dictionary and LLM."""
    
    def __init__(self, openai_api_key: str, dictionary_path: str = "data/btp_dictionary.csv"):
        """
        Initialize the query expansion engine.
        
        Args:
            openai_api_key: OpenAI API key
            dictionary_path: Path to the BTP dictionary CSV file
        """
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        self.dictionary_path = dictionary_path
        self.btp_dictionary = {}
        self.reverse_dictionary = {}
        self.expansion_cache = {}
        self.cache_ttl = timedelta(hours=1)  # Cache expires after 1 hour
        
        # Load dictionary
        self._load_dictionary()
    
    def _load_dictionary(self):
        """Load BTP dictionary from CSV file."""
        try:
            if not Path(self.dictionary_path).exists():
                logger.warning(f"Dictionary file not found: {self.dictionary_path}")
                return
            
            # Read CSV with proper encoding
            df = pd.read_csv(self.dictionary_path, encoding='utf-8')
            
            # Assume columns are: "Mots utilisé par les professionnels du BTP", "Synonyme"
            if len(df.columns) >= 2:
                term_col = df.columns[0]
                synonym_col = df.columns[1]
                
                # Build bidirectional dictionary
                for _, row in df.iterrows():
                    if pd.notna(row[term_col]) and pd.notna(row[synonym_col]):
                        term = str(row[term_col]).strip().lower()
                        synonym = str(row[synonym_col]).strip().lower()
                        
                        # Forward mapping
                        if term not in self.btp_dictionary:
                            self.btp_dictionary[term] = set()
                        self.btp_dictionary[term].add(synonym)
                        
                        # Reverse mapping
                        if synonym not in self.reverse_dictionary:
                            self.reverse_dictionary[synonym] = set()
                        self.reverse_dictionary[synonym].add(term)
                        
                        # Also add bidirectional synonyms
                        if synonym not in self.btp_dictionary:
                            self.btp_dictionary[synonym] = set()
                        self.btp_dictionary[synonym].add(term)
                        
                        if term not in self.reverse_dictionary:
                            self.reverse_dictionary[term] = set()
                        self.reverse_dictionary[term].add(synonym)
            
            logger.info(f"Loaded {len(self.btp_dictionary)} terms from BTP dictionary")
            
        except Exception as e:
            logger.error(f"Error loading dictionary: {str(e)}")
            self.btp_dictionary = {}
            self.reverse_dictionary = {}
    
    def _detect_btp_terms(self, query: str) -> Dict[str, Set[str]]:
        """
        Detect BTP terms in the query and return their synonyms.
        
        Args:
            query: User query
            
        Returns:
            Dictionary mapping detected terms to their synonyms
        """
        detected_terms = {}
        
        # Clean and tokenize query
        query_lower = query.lower()
        # Split on common separators and remove punctuation
        words = re.findall(r'\b\w+\b', query_lower)
        
        # Check each word and common phrases
        for i, word in enumerate(words):
            # Check single words
            if word in self.btp_dictionary:
                detected_terms[word] = self.btp_dictionary[word]
            
            # Check two-word phrases
            if i < len(words) - 1:
                phrase = f"{word} {words[i+1]}"
                if phrase in self.btp_dictionary:
                    detected_terms[phrase] = self.btp_dictionary[phrase]
            
            # Check three-word phrases
            if i < len(words) - 2:
                phrase = f"{word} {words[i+1]} {words[i+2]}"
                if phrase in self.btp_dictionary:
                    detected_terms[phrase] = self.btp_dictionary[phrase]
        
        return detected_terms
    
    def _is_cache_valid(self, cache_entry: Dict) -> bool:
        """Check if cache entry is still valid."""
        if 'timestamp' not in cache_entry:
            return False
        
        cache_time = datetime.fromisoformat(cache_entry['timestamp'])
        return datetime.now() - cache_time < self.cache_ttl
    
    def _get_cached_expansion(self, query: str) -> Optional[List[str]]:
        """Get cached expansion if available and valid."""
        cache_key = query.lower().strip()
        if cache_key in self.expansion_cache:
            cache_entry = self.expansion_cache[cache_key]
            if self._is_cache_valid(cache_entry):
                return cache_entry['expansions']
        return None
    
    def _cache_expansion(self, query: str, expansions: List[str]):
        """Cache expansion results."""
        cache_key = query.lower().strip()
        self.expansion_cache[cache_key] = {
            'expansions': expansions,
            'timestamp': datetime.now().isoformat()
        }
        
        # Limit cache size
        if len(self.expansion_cache) > 100:
            # Remove oldest entries
            sorted_cache = sorted(
                self.expansion_cache.items(),
                key=lambda x: x[1]['timestamp']
            )
            self.expansion_cache = dict(sorted_cache[-100:])
    
    def _generate_llm_expansion(self, query: str, detected_terms: Dict[str, Set[str]]) -> List[str]:
        """
        Generate query expansions using LLM.
        
        Args:
            query: Original query
            detected_terms: Dictionary of detected BTP terms and their synonyms
            
        Returns:
            List of expanded queries
        """
        # Prepare detected terms info for the prompt
        terms_info = ""
        if detected_terms:
            terms_info = "\n**Termes BTP détectés dans la requête:**\n"
            for term, synonyms in detected_terms.items():
                synonyms_str = ", ".join(synonyms)
                terms_info += f"- '{term}' → synonymes: {synonyms_str}\n"
        
        prompt = f"""Tu es un expert en documents techniques BTP. Ta tâche est d'enrichir une requête utilisateur pour améliorer la recherche dans une base de documents.

**Requête originale:** "{query}"

{terms_info}

**Instructions:**
1. Génère 2-3 variantes de la requête qui conservent le sens original
2. Utilise les synonymes BTP détectés quand c'est pertinent
3. Ajoute des termes techniques pertinents de ton expertise BTP
4. Garde les variantes naturelles et recherchables
5. Évite les répétitions exactes

**Format de réponse:**
Retourne uniquement un JSON avec une liste de chaînes, sans explication:
["variante 1", "variante 2", "variante 3"]

**Exemple:**
Requête: "Comment faire un coffrage pour dalle?"
Réponse: ["Comment réaliser une banche pour dalle béton?", "Technique de coffrage pour dalle pleine", "Méthode de mise en œuvre des banches pour plancher"]"""

        # try:
        response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Tu es un expert en terminologie BTP. Réponds uniquement en JSON valide."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        response_text = response.choices[0].message.content.strip()
        print(response_text)
            # Parse JSON response
            # try:
            #     expansions = json.loads(response_text)
            #     if isinstance(expansions, list):
            #         # Filter out empty or too similar expansions
            #         valid_expansions = []
            #         for expansion in expansions:
            #             if (isinstance(expansion, str) and 
            #                 len(expansion.strip()) > 10 and 
            #                 expansion.lower() != query.lower()):
            #                 valid_expansions.append(expansion.strip())
                    
            #         return valid_expansions[:3]  # Limit to 3 expansions
                    
            # except json.JSONDecodeError:
            #     logger.warning(f"Failed to parse JSON response: {response_text}")
            #     # print(response_text)
            #     return []
                
        # except Exception as e:
        #     logger.error(f"Error generating LLM expansion: {str(e)}")
        #     return []
        
        return response_text
    
    def expand_query(self, query: str) -> Dict[str, any]:
        """
        Expand a query using BTP dictionary and LLM.
        
        Args:
            query: Original user query
            
        Returns:
            Dictionary containing:
            - original_query: The original query
            - detected_terms: BTP terms found in query
            - expanded_queries: List of expanded queries
            - expansion_info: Metadata about the expansion
        """
        # Check cache first
        print("Process started")
        cached_expansions = self._get_cached_expansion(query)
        if cached_expansions:
            # Still need to detect terms for metadata
            detected_terms = self._detect_btp_terms(query)
            return {
                "original_query": query,
                "detected_terms": detected_terms,
                "expanded_queries": cached_expansions,
                "expansion_info": {
                    "method": "cached",
                    "terms_detected": len(detected_terms),
                    "expansions_generated": len(cached_expansions)
                }
            }
        
        # Step 1: Detect BTP terms
        detected_terms = self._detect_btp_terms(query)
        
        # Step 2: Generate LLM expansions
        expanded_queries = []
        if len(query.strip()) > 5:  # Only expand meaningful queries
            expanded_queries = self._generate_llm_expansion(query, detected_terms)
            print('Expanded queries:', expanded_queries)
        
        # Cache the results
        if expanded_queries:
            self._cache_expansion(query, expanded_queries)
        
        return {
            "original_query": query,
            # "detected_terms": detected_terms,
            "expanded_queries": expanded_queries,
            # "expansion_info": {
            #     "method": "llm",
            #     "terms_detected": len(detected_terms),
            #     "expansions_generated": len(expanded_queries)
            # }
        }
    
    def get_search_queries(self, query: str, include_original: bool = True) -> List[Dict[str, any]]:
        """
        Get all queries for searching (original + expansions) with weights.
        
        Args:
            query: Original query
            include_original: Whether to include original query
            
        Returns:
            List of query dictionaries with weights
        """
        expansion_result = self.expand_query(query)
        search_queries = []
        
        # Add original query
        if include_original:
            search_queries.append({
                "query": query,
                "weight": 1.0,
                "type": "original"
            })
        
        # Add expanded queries
        for i, expanded_query in enumerate(expansion_result["expanded_queries"]):
            search_queries.append({
                "query": expanded_query,
                "weight": 0.8,  # Lower weight for expanded queries
                "type": "expansion",
                "expansion_rank": i + 1
            })
        
        return search_queries
    
    def get_dictionary_stats(self) -> Dict[str, int]:
        """Get statistics about the loaded dictionary."""
        return {
            "total_terms": len(self.btp_dictionary),
            "cache_size": len(self.expansion_cache),
            "dictionary_loaded": len(self.btp_dictionary) > 0
        }
