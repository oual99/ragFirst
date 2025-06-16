"""Weaviate implementation of the vector database interface."""
import weaviate
import weaviate.classes.config as wc
from weaviate.util import generate_uuid5
from typing import List, Dict, Optional
from tqdm import tqdm
from .vector_db_interface import VectorDBInterface
import config


class WeaviateDatabase(VectorDBInterface):
    """Weaviate implementation of vector database operations."""
    
    def __init__(self, url: str = None, api_key: str = None, openai_api_key: str = None):
        self.url = url or config.WEAVIATE_URL
        self.api_key = api_key or config.WEAVIATE_API_KEY
        self.openai_api_key = openai_api_key or config.OPENAI_API_KEY
        self.client = None
        self.connect()
    
    def connect(self):
        """Connect to Weaviate instance."""
        self.client = weaviate.connect_to_wcs(
            cluster_url=self.url,
            auth_credentials=weaviate.auth.AuthApiKey(self.api_key),
            headers={
                "X-OpenAI-Api-Key": self.openai_api_key
            }
        )
    
    def create_collection(self, collection_name: str) -> bool:
        """Create a new collection in Weaviate."""
        properties = [
            wc.Property(name="source_document", data_type=wc.DataType.TEXT, skip_vectorization=True),
            wc.Property(name="page_number", data_type=wc.DataType.INT, skip_vectorization=True),
            wc.Property(name="paragraph_number", data_type=wc.DataType.INT, skip_vectorization=True),
            wc.Property(name="text", data_type=wc.DataType.TEXT)
        ]
        
        try:
            self.client.collections.create(
                name=collection_name,
                properties=properties,
                vectorizer_config=None
            )
            return True
        except Exception as e:
            print(f"Collection might already exist: {e}")
            return False
    
    def collection_exists(self, collection_name: str) -> bool:
        """Check if a collection exists."""
        try:
            self.client.collections.get(collection_name)
            return True
        except:
            return False
    
    def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection."""
        try:
            self.client.collections.delete(collection_name)
            return True
        except Exception as e:
            print(f"Error deleting collection: {e}")
            return False
    
    def get_collection_stats(self, collection_name: str) -> Dict:
        """Get statistics about a collection."""
        try:
            collection = self.client.collections.get(collection_name)
            # Count objects in collection
            count = 0
            for _ in collection.iterator():
                count += 1
            return {"exists": True, "object_count": count}
        except:
            return {"exists": False, "object_count": 0}
    
    def insert_vectors(self, 
                      collection_name: str, 
                      vectors: List[List[float]], 
                      payloads: List[Dict],
                      ids: Optional[List[str]] = None) -> bool:
        """Insert vectors with payloads into collection."""
        collection = self.client.collections.get(collection_name)
        
        with collection.batch.dynamic() as batch:
            for i, (vector, payload) in enumerate(zip(vectors, payloads)):
                uuid = ids[i] if ids else generate_uuid5(f"{payload['source_document']}_{payload['page_number']}_{payload['paragraph_number']}")
                batch.add_object(
                    properties=payload,
                    uuid=uuid,
                    vector=vector
                )
        
        return len(collection.batch.failed_objects) == 0
    
    def search(self, 
               collection_name: str,
               query_vector: List[float],
               limit: int = 10,
               filters: Optional[Dict] = None) -> List[Dict]:
        """Search for similar vectors."""
        collection = self.client.collections.get(collection_name)
        
        response = collection.query.near_vector(
            near_vector=query_vector,
            limit=limit,
            return_metadata=wc.query.MetadataQuery(distance=True),
            return_properties=[
                "source_document", "page_number", "paragraph_number", "text"
            ]
        )
        
        # Format to match expected structure
        formatted_results = []
        for obj in response.objects:
            result = {
                "distance": obj.metadata.distance if hasattr(obj.metadata, 'distance') else 0.0,
                "properties": obj.properties,
                "metadata": {"distance": obj.metadata.distance if hasattr(obj.metadata, 'distance') else 0.0}
            }
            formatted_results.append(result)
        
        return formatted_results
    
    def ingest_text_data(self, collection_name: str, text_data: List[Dict], 
                        embedding_generator, progress_callback=None):
        """Ingest text data into Weaviate collection with batch embeddings."""
        collection = self.client.collections.get(collection_name)
        
        total_items = len(text_data)
        
        # Extract all texts for batch embedding
        texts = [item['text'] for item in text_data]
        
        # Generate embeddings in batch
        if progress_callback:
            progress_callback(0.3, f"Génération des embeddings pour {len(texts)} textes...")
        
        embeddings = embedding_generator.get_embeddings(texts)
        
        if progress_callback:
            progress_callback(0.6, "Embeddings générés, indexation en cours...")
        
        # Batch insert into Weaviate
        with collection.batch.dynamic() as batch:
            for i, (text_item, embedding) in enumerate(zip(text_data, embeddings)):
                text_obj = {
                    "source_document": text_item['source_document'],
                    "page_number": text_item['page_number'],
                    "paragraph_number": text_item.get('paragraph_number', text_item.get('chunk_index', i)),
                    "text": text_item['text'],
                }
                
                # Add any additional metadata
                for key, value in text_item.items():
                    if key not in text_obj and key not in ['text']:
                        text_obj[key] = value
                
                batch.add_object(
                    properties=text_obj,
                    uuid=generate_uuid5(
                        f"{text_item['source_document']}_{text_item['page_number']}_{text_item.get('paragraph_number', i)}"
                    ),
                    vector=embedding
                )
                
                if progress_callback and i % 10 == 0:
                    progress = 0.6 + (0.4 * (i / total_items))
                    progress_callback(progress, f"Indexation {i+1}/{total_items}...")
        
        if len(collection.batch.failed_objects) > 0:
            print(f"Failed to import {len(collection.batch.failed_objects)} objects")
            return False
        else:
            if progress_callback:
                progress_callback(1.0, "Indexation terminée!")
            print("All objects imported successfully")
            return True
    
    def ingest_text_data_with_progress(self, collection_name: str, text_data: List[Dict], 
                                     embedding_generator, progress_callback=None):
        """Ingest text data with progress reporting."""
        return self.ingest_text_data(collection_name, text_data, embedding_generator, progress_callback)
    
    def close(self):
        """Close the database connection."""
        if self.client:
            self.client.close()