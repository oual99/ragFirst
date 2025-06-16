"""Embeddings generation module."""
from openai import OpenAI
from typing import List, Union
import config


class EmbeddingGenerator:
    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or config.OPENAI_API_KEY
        self.model = model or config.EMBEDDING_MODEL
        self.client = OpenAI(api_key=self.api_key)
        # OpenAI allows up to 2048 inputs per batch
        self.max_batch_size = 100  # Conservative limit for stability
    
    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text (backward compatibility)."""
        return self.get_embeddings([text])[0]
    
    def get_embeddings(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """
        Generate embeddings for one or multiple texts in batch.
        
        Args:
            texts: Single text string or list of texts
            
        Returns:
            List of embedding vectors
        """
        # Handle single text input
        if isinstance(texts, str):
            texts = [texts]
        
        if not texts:
            return []
        
        # Process in batches if needed
        all_embeddings = []
        
        for i in range(0, len(texts), self.max_batch_size):
            batch = texts[i:i + self.max_batch_size]
            
            try:
                response = self.client.embeddings.create(
                    input=batch,
                    model=self.model
                )
                
                # Extract embeddings in order
                batch_embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(batch_embeddings)
                
            except Exception as e:
                # On error, fall back to individual processing for this batch
                print(f"Batch embedding failed: {str(e)}. Falling back to individual processing.")
                for text in batch:
                    try:
                        response = self.client.embeddings.create(
                            input=text,
                            model=self.model
                        )
                        all_embeddings.append(response.data[0].embedding)
                    except Exception as individual_error:
                        print(f"Failed to embed text: {str(individual_error)}")
                        # Return zero vector as fallback
                        all_embeddings.append([0.0] * 3072)  # dimension for text-embedding-3-large
        
        return all_embeddings