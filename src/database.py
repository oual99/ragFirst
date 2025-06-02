"""Database operations module for Weaviate."""
import weaviate
import weaviate.classes.config as wc
from weaviate.util import generate_uuid5
from typing import List, Dict
from tqdm import tqdm
import config


class WeaviateDatabase:
    def __init__(self, url: str = None, api_key: str = None, openai_api_key: str = None):
        self.url = url or config.WEAVIATE_URL
        self.api_key = api_key or config.WEAVIATE_API_KEY
        self.openai_api_key = openai_api_key or config.OPENAI_API_KEY
        self.client = None
        self.connect()
    
    def connect(self):
        """Connect to Weaviate instance."""
        self.client = weaviate.connect_to_weaviate_cloud(
        cluster_url=self.url,
        auth_credentials=weaviate.auth.AuthApiKey(self.api_key),
        headers={
            "X-OpenAI-Api-Key": self.openai_api_key
        }
    )
    
    def create_collection(self, collection_name: str):
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
    
    def ingest_text_data(self, collection_name: str, text_data: List[Dict], embedding_generator):
        """Ingest text data into Weaviate collection."""
        collection = self.client.collections.get(collection_name)
        
        with collection.batch.dynamic() as batch:
            for text in tqdm(text_data, desc="Ingesting text data"):
                vector = embedding_generator.get_embedding(text['text'])
                text_obj = {
                    "source_document": text['source_document'],
                    "page_number": text['page_number'],
                    "paragraph_number": text['paragraph_number'],
                    "text": text['text'],
                }
                batch.add_object(
                    properties=text_obj,
                    uuid=generate_uuid5(f"{text['source_document']}_{text['page_number']}_{text['paragraph_number']}"),
                    vector=vector
                )
        
        if len(collection.batch.failed_objects) > 0:
            print(f"Failed to import {len(collection.batch.failed_objects)} objects")
            return False
        else:
            print("All objects imported successfully")
            return True
    
    def collection_exists(self, collection_name: str) -> bool:
        """Check if a collection exists."""
        try:
            self.client.collections.get(collection_name)
            return True
        except:
            return False
    
    def get_collection_stats(self, collection_name: str) -> dict:
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
    
    def close(self):
        """Close the database connection."""
        if self.client:
            self.client.close()