"""Conversational RAG engine for generating context-aware responses."""
from openai import OpenAI
from typing import List, Dict, Optional
import config


class ConversationalRAGEngine:
    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or config.OPENAI_API_KEY
        self.model = model or config.CHAT_MODEL
        self.client = OpenAI(api_key=self.api_key)
    
    def generate_conversational_response(self, 
                                    query: str, 
                                    search_results: List[Dict],
                                    conversation_history: str = "",
                                    include_sources: bool = True) -> Dict:
        """Generate a conversational response using RAG with conversation context."""
        
        # Prepare context from search results with better source tracking
        document_context = ""
        source_mapping = {}  # Map document numbers to actual names
        
        if search_results:
            for i, item in enumerate(search_results):
                doc_name = item['source_document']
                # Remove file extension for cleaner reference
                clean_doc_name = doc_name.replace('.pdf', '').replace('.PDF', '')
                
                document_context += (
                    f"[Source {i+1}] {doc_name} "
                    f"(Page {item['page_number']}, Chunk {item.get('paragraph_number', item.get('chunk_index', ''))}): "
                    f"{item['text']}\n\n"
                )
                
                # Store mapping for reference
                source_mapping[f"Source {i+1}"] = {
                    "full_name": doc_name,
                    "clean_name": clean_doc_name,
                    "page": item['page_number'],
                    "chunk": item.get('paragraph_number', item.get('chunk_index', ''))
                }
        
        # Enhanced system prompt with better citation instructions
        prompt_system = (
            "Tu es un assistant IA spécialisé dans l'analyse de documents BTP (Bâtiment et Travaux Publics). "
            "Tu es poli, amical et professionnel.\n\n"
            "RÈGLES IMPORTANTES À SUIVRE:\n\n"
            "1. CITATIONS DES SOURCES:\n"
            "   - TOUJOURS citer les sources en utilisant le nom EXACT du document\n"
            "   - Format: (NomDuDocument.pdf, Page X)\n"
            "   - NE PAS utiliser 'Document 1' ou 'Source 1' - utilise le VRAI nom du fichier\n"
            "   - Si tu cites plusieurs fois le même document, répète son nom complet\n\n"
            "2. DÉTECTION DES CONTRADICTIONS:\n"
            "   - SEULEMENT si tu trouves des informations contradictoires, tu dois le signaler\n"
            "   - S'il n'y a PAS de contradiction, réponds DIRECTEMENT sans mentionner l'absence de contradiction\n"
            "   - Format pour les contradictions: 'J'ai trouvé des informations contradictoires concernant [sujet]:\n"
            "     • Dans NomDocument1.pdf, Page Y: [information 1]\n"
            "     • Dans NomDocument2.pdf, Page W: [information 2]'\n\n"
            "3. RÉPONSES NORMALES (sans contradiction):\n"
            "   - Donne l'information directement avec la source complète\n"
            "   - Exemple: 'Le montant du marché est de 13 490 000 € HT (CCAP_Travaux.pdf, Page 71).'\n"
            "   - NE DIS PAS: 'je n'ai pas trouvé d'autres informations qui contredisent'\n"
            "   - NE DIS PAS: 'Cependant, je n'ai pas trouvé...'\n\n"
            "4. ANALYSE DES RÉPONSES:\n"
            "   - Vérifie s'il y a des incohérences SEULEMENT si plusieurs sources parlent du même sujet\n"
            "   - Une seule source = pas de mention de contradiction\n"
            "   - Plusieurs sources concordantes = cite-les toutes simplement\n"
            "   - Plusieurs sources contradictoires = signale la contradiction\n\n"
            "5. TYPES DE QUESTIONS:\n"
            "   - Salutations: Réponds amicalement\n"
            "   - Questions factuelles: Utilise UNIQUEMENT les documents\n"
            "   - Si aucune info trouvée: 'Je n'ai pas trouvé cette information dans les documents fournis.'\n"
            "   - Questions hors BTP: Redirige poliment vers le domaine BTP\n\n"
            "6. STYLE DE RÉPONSE:\n"
            "   - Sois concis et direct\n"
            "   - Cite tes sources avec le nom complet du document\n"
            "   - N'ajoute pas de phrases inutiles sur ce que tu n'as pas trouvé"
        )
        
        # Construct the user prompt with source mapping
        prompt_parts = []
        
        if conversation_history:
            prompt_parts.append(f"Historique de la conversation:\n{conversation_history}\n")
        
        if document_context:
            prompt_parts.append(f"Contenu des documents trouvés:\n{document_context}")
            
            # Add source mapping information
            prompt_parts.append("\nMAPPING DES SOURCES (utilise ces noms dans tes citations):")
            for source_ref, info in source_mapping.items():
                prompt_parts.append(f"{source_ref} = {info['full_name']} (Page {info['page']})")
                
        elif search_results is not None and len(search_results) == 0:
            prompt_parts.append("Note: Aucun document pertinent trouvé pour cette recherche.")
        
        prompt_parts.append(f"\nQuestion de l'utilisateur: {query}")
        
        prompt_parts.append(
            "\nInstructions pour répondre:\n"
            "1. Utilise TOUJOURS le nom complet du document dans tes citations (pas 'Source 1' ou 'Document 1')\n"
            "2. Vérifie s'il y a des informations contradictoires SEULEMENT si tu as plusieurs sources sur le même sujet\n"
            "3. Si une seule source ou pas de contradiction: réponds directement avec l'information et la source\n"
            "4. Si contradiction détectée: commence par signaler la contradiction\n"
            "5. Exemples de bonnes citations:\n"
            "   - CORRECT: 'charge de 1000 KN/m² (01_Gros-Oeuvre_VSS.pdf, Page 9)'\n"
            "   - INCORRECT: 'charge de 1000 KN/m² (Document 1, Page 9)'\n"
            "   - INCORRECT: 'charge de 1000 KN/m² (Source 1, Page 9)'"
        )
        
        prompt_user = "\n\n".join(prompt_parts)
        
        # Generate response
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": prompt_system},
                {"role": "user", "content": prompt_user}
            ],
            temperature=0.1,  # Low temperature for accurate fact reporting
            max_tokens=1000
        )
        
        ai_response = response.choices[0].message.content
        
        # Post-process response to ensure correct citations
        # Replace any remaining "Source X" references with actual document names
        for source_ref, info in source_mapping.items():
            # Replace various possible formats
            ai_response = ai_response.replace(f"({source_ref},", f"({info['full_name']},")
            ai_response = ai_response.replace(f"[{source_ref}]", f"[{info['full_name']}]")
            ai_response = ai_response.replace(f"{source_ref} ", f"{info['full_name']} ")
        
        # Check if contradictions were found (for UI enhancement)
        has_contradictions = any(word in ai_response.lower() for word in 
                            ['contradiction', 'contradictoire', 'incohérent', 'différent'])
        
        # Generate follow-up suggestions
        follow_up_suggestions = []
        if search_results:
            # If contradictions found, suggest clarification questions
            if has_contradictions:
                follow_up_suggestions = [
                    "Quelle est la source la plus récente?",
                    "Y a-t-il d'autres documents qui pourraient clarifier?",
                    "Ces différences sont-elles significatives pour le projet?"
                ]
            else:
                follow_up_suggestions = self._generate_follow_up_suggestions(query, ai_response)
        
        # Format sources if needed
        sources = []
        if include_sources and search_results:
            for item in search_results:
                source = {
                    "type": "text",
                    "distance": item.get('distance', 0),
                    "document": item['source_document'],
                    "page": item['page_number'],
                    "paragraph": item.get('paragraph_number', item.get('chunk_index', ''))
                }
                sources.append(source)
            sources.sort(key=lambda x: x['distance'])
        
        return {
            "response": ai_response,
            "sources": sources,
            "follow_up_suggestions": follow_up_suggestions,
            "has_contradictions": has_contradictions
        }
    
    def _generate_follow_up_suggestions(self, query: str, response: str) -> List[str]:
        """Generate contextual follow-up question suggestions."""
        prompt = (
            f"Basé sur cette question: '{query}' et cette réponse: '{response}', "
            "suggère 3 questions de suivi courtes et pertinentes que l'utilisateur pourrait poser. "
            "Les questions doivent être en français, naturelles et directement liées au contexte BTP. "
            "Format: une question par ligne, sans numérotation."
        )
        
        try:
            suggestion_response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Tu es un assistant qui génère des questions de suivi pertinentes."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,
                max_tokens=150
            )
            
            suggestions = suggestion_response.choices[0].message.content.strip().split('\n')
            # Clean and filter suggestions
            suggestions = [s.strip() for s in suggestions if s.strip() and len(s.strip()) > 10][:3]
            
            return suggestions
        except:
            # Return empty list if generation fails
            return []
    
    def analyze_intent(self, query: str, conversation_history: str = "") -> Dict:
        """Analyze user intent to determine the type of response needed."""
        prompt = (
            f"Analyse cette question dans le contexte d'une conversation sur des documents BTP:\n"
            f"Historique: {conversation_history[-500:] if conversation_history else 'Début de conversation'}\n"
            f"Question: {query}\n\n"
            "Détermine:\n"
            "1. intent: 'search' (recherche de documents), 'clarification' (clarification), "
            "'follow_up' (question de suivi), 'greeting' (salutation), 'thanks' (remerciement)\n"
            "2. requires_new_search: true/false\n"
            "3. confidence: 0-1\n"
            "Format JSON"
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Tu analyses les intentions des utilisateurs. Réponds uniquement en JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=100
            )
            
            import json
            return json.loads(response.choices[0].message.content)
        except:
            # Default intent analysis
            return {
                "intent": "search",
                "requires_new_search": True,
                "confidence": 0.5
            }
    
    def summarize_conversation(self, messages: List[Dict], max_length: int = 500) -> str:
        """Summarize a conversation to fit within token limits."""
        if not messages:
            return ""
        
        # Convert messages to text
        conversation_text = "\n".join([
            f"{msg['role'].capitalize()}: {msg['content'][:200]}..." 
            if len(msg['content']) > 200 else f"{msg['role'].capitalize()}: {msg['content']}"
            for msg in messages
        ])
        
        if len(conversation_text) <= max_length:
            return conversation_text
        
        # Summarize if too long
        prompt = (
            f"Résume cette conversation en gardant les points clés et le contexte important. "
            f"Maximum {max_length} caractères:\n\n{conversation_text}"
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Tu résumes des conversations de manière concise."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=max_length // 4  # Rough token to character ratio
            )
            
            return response.choices[0].message.content
        except:
            # Return truncated version if summarization fails
            return conversation_text[:max_length] + "..."
    
    
    
    # In src/conversational_rag_engine.py, add these methods to the ConversationalRAGEngine class
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the assistant."""
        return (
            "Tu es un assistant IA spécialisé dans l'analyse de documents BTP (Bâtiment et Travaux Publics). "
            "Tu es poli, amical et professionnel.\n\n"
            "RÈGLES IMPORTANTES À SUIVRE:\n\n"
            "1. CITATIONS DES SOURCES:\n"
            "   - TOUJOURS citer les sources en utilisant le nom EXACT du document\n"
            "   - Format: (NomDuDocument.pdf, Page X)\n"
            "   - NE PAS utiliser 'Document 1' ou 'Source 1' - utilise le VRAI nom du fichier\n"
            "   - Si tu cites plusieurs fois le même document, répète son nom complet\n\n"
            "2. DÉTECTION DES CONTRADICTIONS:\n"
            "   - SEULEMENT si tu trouves des informations contradictoires, tu dois le signaler\n"
            "   - S'il n'y a PAS de contradiction, réponds DIRECTEMENT sans mentionner l'absence de contradiction\n"
            "   - Format pour les contradictions: 'J'ai trouvé des informations contradictoires concernant [sujet]:\n"
            "     • Dans NomDocument1.pdf, Page Y: [information 1]\n"
            "     • Dans NomDocument2.pdf, Page W: [information 2]'\n\n"
            "3. RÉPONSES NORMALES (sans contradiction):\n"
            "   - Donne l'information directement avec la source complète\n"
            "   - Exemple: 'Le montant du marché est de 13 490 000 € HT (CCAP_Travaux.pdf, Page 71).'\n"
            "   - NE DIS PAS: 'je n'ai pas trouvé d'autres informations qui contredisent'\n"
            "   - NE DIS PAS: 'Cependant, je n'ai pas trouvé...'\n\n"
            "4. ANALYSE DES RÉPONSES:\n"
            "   - Vérifie s'il y a des incohérences SEULEMENT si plusieurs sources parlent du même sujet\n"
            "   - Une seule source = pas de mention de contradiction\n"
            "   - Plusieurs sources concordantes = cite-les toutes simplement\n"
            "   - Plusieurs sources contradictoires = signale la contradiction\n\n"
            "5. TYPES DE QUESTIONS:\n"
            "   - Salutations: Réponds amicalement\n"
            "   - Questions factuelles: Utilise UNIQUEMENT les documents\n"
            "   - Si aucune info trouvée: 'Je n'ai pas trouvé cette information dans les documents fournis.'\n"
            "   - Questions hors BTP: Redirige poliment vers le domaine BTP\n\n"
            "6. STYLE DE RÉPONSE:\n"
            "   - Sois concis et direct\n"
            "   - Cite tes sources avec le nom complet du document\n"
            "   - N'ajoute pas de phrases inutiles sur ce que tu n'as pas trouvé"
        )

    def _build_user_prompt(self, query: str, conversation_history: str, 
                        document_context: str, source_mapping: Dict) -> str:
        """Build the user prompt for the LLM."""
        prompt_parts = []
        
        if conversation_history:
            prompt_parts.append(f"Historique de la conversation:\n{conversation_history}\n")
        
        if document_context:
            prompt_parts.append(f"Contenu des documents trouvés:\n{document_context}")
            
            # Add source mapping information
            prompt_parts.append("\nMAPPING DES SOURCES (utilise ces noms dans tes citations):")
            for source_ref, info in source_mapping.items():
                prompt_parts.append(f"{source_ref} = {info['full_name']} (Page {info['page']})")
                
        elif document_context is not None:
            prompt_parts.append("Note: Aucun document pertinent trouvé pour cette recherche.")
        
        prompt_parts.append(f"\nQuestion de l'utilisateur: {query}")
        
        prompt_parts.append(
            "\nInstructions pour répondre:\n"
            "1. Utilise TOUJOURS le nom complet du document dans tes citations (pas 'Source 1' ou 'Document 1')\n"
            "2. Vérifie s'il y a des informations contradictoires SEULEMENT si tu as plusieurs sources sur le même sujet\n"
            "3. Si une seule source ou pas de contradiction: réponds directement avec l'information et la source\n"
            "4. Si contradiction détectée: commence par signaler la contradiction\n"
            "5. Exemples de bonnes citations:\n"
            "   - CORRECT: 'charge de 1000 KN/m² (01_Gros-Oeuvre_VSS.pdf, Page 9)'\n"
            "   - INCORRECT: 'charge de 1000 KN/m² (Document 1, Page 9)'\n"
            "   - INCORRECT: 'charge de 1000 KN/m² (Source 1, Page 9)'"
        )
        
        return "\n\n".join(prompt_parts)

    def _format_sources(self, search_results: List[Dict]) -> List[Dict]:
        """Format sources for metadata."""
        sources = []
        if search_results:
            for item in search_results:
                source = {
                    "type": "text",
                    "distance": item.get('distance', 0),
                    "document": item['source_document'],
                    "page": item['page_number'],
                    "paragraph": item.get('paragraph_number', item.get('chunk_index', ''))
                }
                sources.append(source)
            sources.sort(key=lambda x: x['distance'])
        return sources

    def generate_conversational_response_stream(self, 
                                            query: str, 
                                            search_results: List[Dict],
                                            conversation_history: str = "",
                                            include_sources: bool = True):
        """Generate a streaming conversational response."""
        
        # Prepare context
        document_context = ""
        source_mapping = {}
        
        if search_results:
            for i, item in enumerate(search_results):
                doc_name = item['source_document']
                clean_doc_name = doc_name.replace('.pdf', '').replace('.PDF', '')
                
                document_context += (
                    f"[Source {i+1}] {doc_name} "
                    f"(Page {item['page_number']}, Chunk {item.get('paragraph_number', item.get('chunk_index', ''))}): "
                    f"{item['text']}\n\n"
                )
                
                source_mapping[f"Source {i+1}"] = {
                    "full_name": doc_name,
                    "clean_name": clean_doc_name,
                    "page": item['page_number'],
                    "chunk": item.get('paragraph_number', item.get('chunk_index', ''))
                }
        
        # Get prompts
        prompt_system = self._get_system_prompt()
        prompt_user = self._build_user_prompt(query, conversation_history, document_context, source_mapping)
        
        # Create streaming response
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": prompt_system},
                {"role": "user", "content": prompt_user}
            ],
            temperature=0.1,
            max_tokens=1000,
            stream=True
        )
        
        # Yield chunks with metadata
        full_response = ""
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                full_response += content
                
                # Fix source references on the fly
                for source_ref, info in source_mapping.items():
                    content = content.replace(f"({source_ref},", f"({info['full_name']},")
                    content = content.replace(f"[{source_ref}]", f"[{info['full_name']}]")
                    content = content.replace(f"{source_ref} ", f"{info['full_name']} ")
                
                yield {
                    "type": "content",
                    "content": content
                }
        
        # After streaming completes, yield metadata
        has_contradictions = any(word in full_response.lower() for word in 
                            ['contradiction', 'contradictoire', 'incohérent', 'différent'])
        
        # Generate follow-up suggestions
        follow_up_suggestions = []
        if search_results:
            if has_contradictions:
                follow_up_suggestions = [
                    "Quelle est la source la plus récente?",
                    "Y a-t-il d'autres documents qui pourraient clarifier?",
                    "Ces différences sont-elles significatives pour le projet?"
                ]
            else:
                # For streaming, skip follow-up generation to avoid delay
                follow_up_suggestions = []
        
        # Yield final metadata
        yield {
            "type": "metadata",
            "has_contradictions": has_contradictions,
            "follow_up_suggestions": follow_up_suggestions,
            "sources": self._format_sources(search_results) if include_sources else [],
            "full_response": full_response
        }