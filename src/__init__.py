"""Initialize the src package with all modules."""

from .document_processor import DocumentProcessor
from .embeddings import EmbeddingGenerator
from .database import WeaviateDatabase
from .search import SearchEngine
from .rag_engine import RAGEngine
from .conversational_rag_engine import ConversationalRAGEngine
from .conversation_manager import ConversationManager
from .vector_db_interface import VectorDBInterface
from .qdrant_database import QdrantDatabase

__all__ = [
    'DocumentProcessor',
    'EmbeddingGenerator',
    'WeaviateDatabase',
    'SearchEngine',
    'RAGEngine',
    'ConversationalRAGEngine',
    'ConversationManager'
]

__version__ = '2.0.0'  # Updated for conversational support