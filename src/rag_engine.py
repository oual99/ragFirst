"""RAG engine for generating responses."""
from openai import OpenAI
from typing import List, Dict
import config


class RAGEngine:
    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or config.OPENAI_API_KEY
        self.model = model or config.CHAT_MODEL
        self.client = OpenAI(api_key=self.api_key)
    
    def generate_response(self, query: str, context: str) -> str:
        """Generate response using RAG."""
        prompt_system = (
            "Tu es un assistant IA spécialisé dans le secteur BTP (Bâtiment et Travaux Publics). "
            "Tu dois uniquement utiliser les informations fournies pour répondre. "
            "Si elles sont insuffisantes, indique clairement : "
            "« D'après les documents fournis, je ne peux pas répondre précisément à cette question. »"
        )
        
        prompt_user = (
            f"Contexte (résumés/documents récupérés) :\n{context}\n\n"
            f"Question de l'utilisateur : {query}\n\n"
            "Veuillez fournir une réponse détaillée, précise et strictement fondée sur ce contexte. "
            "Si tu cites un extrait, mentionne l'ID du document entre crochets (ex. [doc_2])."
        )
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": prompt_system},
                {"role": "user", "content": prompt_user}
            ],
            temperature=0
        )
        
        return response.choices[0].message.content
    
    def analyze(self, user_query: str, search_results: List[Dict]) -> Dict:
        """Perform complete RAG analysis."""
        # Prepare context for RAG
        context = ""
        for item in search_results:
            context += (
                f"Text from {item['source_document']} "
                f"(Page {item['page_number']}, Paragraph {item['paragraph_number']}): "
                f"{item['text']}\n\n"
            )
        
        # Generate response using RAG
        response = self.generate_response(user_query, context)
        
        # Format sources
        sources = []
        for item in search_results:
            source = {
                "type": "text",
                "distance": item['distance'],
                "document": item['source_document'],
                "page": item['page_number'],
                "paragraph": item['paragraph_number']
            }
            sources.append(source)
        
        # Sort sources by distance (ascending order)
        sources.sort(key=lambda x: x['distance'])
        
        return {
            "user_query": user_query,
            "ai_response": response,
            "sources": sources
        }