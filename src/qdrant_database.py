"""Qdrant implementation of the vector database interface."""
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from typing import List, Dict, Optional
import uuid
from tqdm import tqdm
from .vector_db_interface import VectorDBInterface


class QdrantDatabase(VectorDBInterface):
    """Qdrant implementation of vector database operations."""
    
    def __init__(self, url: str = None, api_key: str = None, **kwargs):
        self.url = url
        self.api_key = api_key
        self.client = None
        self.vector_size = 3072  # Default for text-embedding-3-large
        self.connect()
    
    def connect(self):
        """Connect to Qdrant instance."""
        self.client = QdrantClient(
            url=self.url,
            api_key=self.api_key,
            timeout=30
        )
    
    def create_collection(self, collection_name: str) -> bool:
        """Create a new collection in Qdrant."""
        try:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE
                )
            )
            return True
        except Exception as e:
            print(f"Collection might already exist: {e}")
            return False
    
    def collection_exists(self, collection_name: str) -> bool:
        """Check if a collection exists."""
        try:
            collections = self.client.get_collections().collections
            return any(col.name == collection_name for col in collections)
        except:
            return False
    
    def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection."""
        try:
            self.client.delete_collection(collection_name=collection_name)
            return True
        except Exception as e:
            print(f"Error deleting collection: {e}")
            return False
    
    def get_collection_stats(self, collection_name: str) -> Dict:
        """Get statistics about a collection."""
        try:
            if self.collection_exists(collection_name):
                info = self.client.get_collection(collection_name=collection_name)
                return {
                    "exists": True,
                    "object_count": info.points_count,
                    "vectors_count": info.vectors_count
                }
            return {"exists": False, "object_count": 0}
        except:
            return {"exists": False, "object_count": 0}
    
    def insert_vectors(self, 
                      collection_name: str, 
                      vectors: List[List[float]], 
                      payloads: List[Dict],
                      ids: Optional[List[str]] = None) -> bool:
        """Insert vectors with payloads into collection."""
        try:
            if ids is None:
                ids = [str(uuid.uuid4()) for _ in range(len(vectors))]
            
            points = [
                PointStruct(
                    id=id_,
                    vector=vector,
                    payload=payload
                )
                for id_, vector, payload in zip(ids, vectors, payloads)
            ]
            
            # Batch upload for better performance
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                self.client.upsert(
                    collection_name=collection_name,
                    points=batch
                )
            
            return True
        except Exception as e:
            print(f"Error inserting vectors: {e}")
            return False
    
    def search(self, 
               collection_name: str,
               query_vector: List[float],
               limit: int = 10,
               filters: Optional[Dict] = None) -> List[Dict]:
        """Search for similar vectors."""
        try:
            results = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                query_filter=filters,
                with_payload=True,
                with_vectors=False
            )
            
            # Format results to match expected structure
            formatted_results = []
            for hit in results:
                # Create a result object that mimics Weaviate's structure
                class ResultObject:
                    def __init__(self, properties, metadata):
                        self.properties = properties
                        self.metadata = metadata
                
                # Create metadata object with distance
                metadata = type('obj', (object,), {
                    'distance': 1 - hit.score  # Convert similarity to distance
                })
                
                result_obj = ResultObject(
                    properties=hit.payload,
                    metadata=metadata
                )
                
                formatted_results.append(result_obj)
            
            return formatted_results
        except Exception as e:
            print(f"Error searching: {e}")
            return []
    
    def ingest_text_data(self,
                        collection_name: str,
                        text_data: List[Dict],
                        embedding_generator,
                        progress_callback=None) -> bool:
        """Ingest text data with batch embeddings."""
        try:
            # Extract all texts
            texts = [item['text'] for item in text_data]
            total_items = len(texts)
            
            # Generate embeddings in batch
            if progress_callback:
                progress_callback(0.3, f"Génération des embeddings pour {total_items} textes...")
            
            embeddings = embedding_generator.get_embeddings(texts)
            
            if progress_callback:
                progress_callback(0.6, "Embeddings générés, indexation en cours...")
            
            # Prepare batch data
            ids = []
            payloads = []
            
            for i, text_item in enumerate(text_data):
                # Generate ID
                doc_id = f"{text_item['source_document']}_{text_item['page_number']}_{text_item.get('paragraph_number', i)}"
                ids.append(str(uuid.uuid5(uuid.NAMESPACE_DNS, doc_id)))
                
                # Prepare payload
                payload = {
                    "source_document": text_item['source_document'],
                    "page_number": text_item['page_number'],
                    "paragraph_number": text_item.get('paragraph_number', text_item.get('chunk_index', i)),
                    "text": text_item['text']
                }
                
                # Add any additional metadata
                for key, value in text_item.items():
                    if key not in payload and key != 'text':
                        payload[key] = value
                
                payloads.append(payload)
            
            # Batch insert
            success = self.insert_vectors(
                collection_name=collection_name,
                vectors=embeddings,
                payloads=payloads,
                ids=ids
            )
            
            if progress_callback:
                progress_callback(1.0, "Indexation terminée!")
            
            return success
            
        except Exception as e:
            print(f"Failed to ingest data: {e}")
            return False
    
    def ingest_text_data_with_progress(self,
                                     collection_name: str,
                                     text_data: List[Dict],
                                     embedding_generator,
                                     progress_callback=None) -> bool:
        """Ingest text data with progress reporting (same as ingest_text_data for Qdrant)."""
        return self.ingest_text_data(collection_name, text_data, embedding_generator, progress_callback)
    
    def close(self):
        """Close the database connection (no-op for Qdrant HTTP client)."""
        pass