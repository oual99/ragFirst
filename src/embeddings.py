"""Embeddings generation module."""
from openai import OpenAI
from typing import List
import config


class EmbeddingGenerator:
    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or config.OPENAI_API_KEY
        self.model = model or config.EMBEDDING_MODEL
        self.client = OpenAI(api_key=self.api_key)
    
    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding for a given text."""
        response = self.client.embeddings.create(
            input=text,
            model=self.model
        )
        return response.data[0].embedding