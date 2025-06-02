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
        
        # Prepare context from search results
        document_context = ""
        if search_results:
            for i, item in enumerate(search_results):
                document_context += (
                    f"[Document {i+1}] {item['source_document']} "
                    f"(Page {item['page_number']}, Paragraphe {item['paragraph_number']}): "
                    f"{item['text']}\n\n"
                )
        
        # System prompt for conversational AI
        prompt_system = (
            "Tu es un assistant conversationnel IA spécialisé dans le secteur BTP (Bâtiment et Travaux Publics). "
            "Tu maintiens une conversation naturelle et fluide tout en étant précis et informatif. "
            "Tu te souviens du contexte de la conversation et tu peux faire référence aux échanges précédents. "
            "Utilise un ton professionnel mais amical. "
            "Si les informations fournies sont insuffisantes, propose de chercher plus de détails ou pose des questions clarificatrices. "
            "Évite de répéter les mêmes informations sauf si explicitement demandé."
        )
        
        # Construct the user prompt
        prompt_parts = []
        
        if conversation_history:
            prompt_parts.append(f"Historique de la conversation:\n{conversation_history}\n")
        
        if document_context:
            prompt_parts.append(f"Documents pertinents trouvés:\n{document_context}")
        elif search_results is not None and len(search_results) == 0:
            prompt_parts.append("Aucun document pertinent n'a été trouvé pour cette question.")
        
        prompt_parts.append(f"Question actuelle de l'utilisateur: {query}")
        
        prompt_parts.append(
            "\nRéponds de manière conversationnelle en tenant compte du contexte. "
            "Si tu cites des documents, mentionne-les naturellement dans ta réponse."
        )
        
        prompt_user = "\n\n".join(prompt_parts)
        
        # Generate response
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": prompt_system},
                {"role": "user", "content": prompt_user}
            ],
            temperature=0.5,  # Slightly higher for more natural conversation
            max_tokens=1000
        )
        
        ai_response = response.choices[0].message.content
        
        # Generate follow-up suggestions
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
                    "paragraph": item['paragraph_number']
                }
                sources.append(source)
            sources.sort(key=lambda x: x['distance'])
        
        return {
            "response": ai_response,
            "sources": sources,
            "follow_up_suggestions": follow_up_suggestions
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