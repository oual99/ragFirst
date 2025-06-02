"""Conversation management module for handling chat history and context."""
import json
import os
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from collections import deque


class ConversationManager:
    def __init__(self, max_history_length: int = 20, conversation_dir: str = "data/conversations"):
        self.max_history_length = max_history_length
        self.conversation_dir = conversation_dir
        os.makedirs(conversation_dir, exist_ok=True)
    
    def get_formatted_history(self, messages: List[Dict], max_chars: int = 2000) -> str:
        """Format conversation history for inclusion in prompts."""
        if not messages:
            return ""
        
        formatted_messages = []
        total_chars = 0
        
        # Process messages in reverse order (most recent first)
        for message in reversed(messages):
            if message['role'] == 'system':
                continue
                
            # Format message
            timestamp = datetime.fromisoformat(message['timestamp']).strftime('%H:%M')
            role = "Utilisateur" if message['role'] == 'user' else "Assistant"
            content = message['content']
            
            # Truncate long messages
            if len(content) > 300:
                content = content[:297] + "..."
            
            formatted_msg = f"[{timestamp}] {role}: {content}"
            
            # Check if adding this message would exceed limit
            if total_chars + len(formatted_msg) > max_chars:
                break
            
            formatted_messages.append(formatted_msg)
            total_chars += len(formatted_msg)
        
        # Return in chronological order
        return "\n".join(reversed(formatted_messages))
    
    def save_conversation(self, conversation_id: str, messages: List[Dict]) -> str:
        """Save conversation to disk."""
        filename = f"{conversation_id}.json"
        filepath = os.path.join(self.conversation_dir, filename)
        
        conversation_data = {
            "conversation_id": conversation_id,
            "created_at": messages[0]['timestamp'] if messages else datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "message_count": len(messages),
            "messages": messages
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(conversation_data, f, ensure_ascii=False, indent=2)
        
        return filepath
    
    def load_conversation(self, conversation_id: str) -> Optional[Dict]:
        """Load conversation from disk."""
        filename = f"{conversation_id}.json"
        filepath = os.path.join(self.conversation_dir, filename)
        
        if not os.path.exists(filepath):
            return None
        
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def list_conversations(self, days: int = 7) -> List[Dict]:
        """List recent conversations."""
        conversations = []
        cutoff_date = datetime.now() - timedelta(days=days)
        
        for filename in os.listdir(self.conversation_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(self.conversation_dir, filename)
                
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    last_updated = datetime.fromisoformat(data.get('last_updated', ''))
                    
                    if last_updated > cutoff_date:
                        conversations.append({
                            'conversation_id': data['conversation_id'],
                            'created_at': data.get('created_at'),
                            'last_updated': data.get('last_updated'),
                            'message_count': data.get('message_count', 0),
                            'preview': data['messages'][0]['content'][:100] if data.get('messages') else ''
                        })
                except:
                    continue
        
        # Sort by last updated, most recent first
        conversations.sort(key=lambda x: x['last_updated'], reverse=True)
        return conversations
    
    def extract_key_points(self, messages: List[Dict]) -> List[str]:
        """Extract key points from conversation for context."""
        key_points = []
        
        # Look for messages with specific patterns
        patterns = {
            'questions': ['quel', 'comment', 'où', 'quand', 'combien', 'pourquoi'],
            'technical_terms': ['ascenseur', 'bâtiment', 'étage', 'niveau', 'construction', 
                              'structure', 'dimension', 'matériau', 'norme', 'sécurité'],
            'numbers': ['m²', 'mètres', 'cm', 'mm', '%', '€']
        }
        
        for message in messages:
            if message['role'] == 'user':
                content_lower = message['content'].lower()
                
                # Check for important patterns
                is_important = False
                for category, terms in patterns.items():
                    if any(term in content_lower for term in terms):
                        is_important = True
                        break
                
                if is_important:
                    key_points.append(f"Question: {message['content'][:150]}")
            
            elif message['role'] == 'assistant' and 'sources' in message:
                # If assistant cited sources, it's likely important
                key_points.append(f"Réponse documentée: {message['content'][:150]}")
        
        return key_points[-5:]  # Keep last 5 key points
    
    def get_conversation_context(self, messages: List[Dict]) -> Dict:
        """Get comprehensive conversation context."""
        if not messages:
            return {
                "summary": "",
                "key_points": [],
                "mentioned_documents": [],
                "topics": []
            }
        
        # Extract mentioned documents
        mentioned_documents = set()
        topics = set()
        
        for message in messages:
            if 'sources' in message:
                for source in message['sources']:
                    mentioned_documents.add(os.path.basename(source['document']))
            
            # Extract potential topics (simplified)
            content_lower = message['content'].lower()
            if 'ascenseur' in content_lower:
                topics.add('ascenseurs')
            if 'bâtiment' in content_lower:
                topics.add('bâtiments')
            if 'sécurité' in content_lower:
                topics.add('sécurité')
            if 'norme' in content_lower or 'réglementation' in content_lower:
                topics.add('normes et réglementations')
        
        return {
            "summary": self.get_formatted_history(messages[-10:], max_chars=500),
            "key_points": self.extract_key_points(messages),
            "mentioned_documents": list(mentioned_documents),
            "topics": list(topics)
        }
    
    def should_summarize_context(self, messages: List[Dict]) -> bool:
        """Determine if conversation context should be summarized."""
        if len(messages) < 10:
            return False
        
        # Calculate total character count of last 10 messages
        total_chars = sum(len(msg['content']) for msg in messages[-10:])
        
        return total_chars > 3000
    
    def manage_conversation_memory(self, messages: List[Dict]) -> List[Dict]:
        """Manage conversation memory to prevent context overflow."""
        if len(messages) <= self.max_history_length:
            return messages
        
        # Keep system messages and recent messages
        system_messages = [msg for msg in messages if msg['role'] == 'system']
        recent_messages = messages[-self.max_history_length:]
        
        # Combine, ensuring no duplicates
        managed_messages = system_messages + [
            msg for msg in recent_messages if msg not in system_messages
        ]
        
        return managed_messages