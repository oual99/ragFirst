from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import json


class VectorDBInterface(ABC):
    """Abstract base class for vector database operations."""
    
    @abstractmethod
    def connect(self):
        """Connect to the vector database."""
        pass
    
    @abstractmethod
    def create_collection(self, collection_name: str) -> bool:
        """Create a new collection."""
        pass
    
    @abstractmethod
    def collection_exists(self, collection_name: str) -> bool:
        """Check if a collection exists."""
        pass
    
    @abstractmethod
    def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection."""
        pass
    
    @abstractmethod
    def get_collection_stats(self, collection_name: str) -> Dict:
        """Get statistics about a collection."""
        pass
    
    @abstractmethod
    def insert_vectors(self, 
                      collection_name: str, 
                      vectors: List[List[float]], 
                      payloads: List[Dict],
                      ids: Optional[List[str]] = None) -> bool:
        """Insert vectors with payloads into collection."""
        pass
    
    @abstractmethod
    def search(self, 
               collection_name: str,
               query_vector: List[float],
               limit: int = 10,
               filters: Optional[Dict] = None) -> List[Dict]:
        """Search for similar vectors."""
        pass
    
    @abstractmethod
    def ingest_text_data(self,
                        collection_name: str,
                        text_data: List[Dict],
                        embedding_generator,
                        progress_callback=None) -> bool:
        """Ingest text data with embeddings."""
        pass
    
    @abstractmethod
    def close(self):
        """Close the database connection."""
        pass