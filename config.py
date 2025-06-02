"""Configuration file for the RAG system."""
import os
from dotenv import load_dotenv
import streamlit as st


# Load environment variables
load_dotenv()


# Weaviate Configuration
# Use Streamlit secrets in production
if "WEAVIATE_URL" in st.secrets:
    WEAVIATE_URL = st.secrets["WEAVIATE_URL"]
    WEAVIATE_API_KEY = st.secrets["WEAVIATE_API_KEY"]
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
else:
    WEAVIATE_URL = os.getenv("WEAVIATE_URL", "")
    WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY", "")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Collection Configuration
COLLECTION_NAME = "RAGESGDocuments"

# Model Configuration
EMBEDDING_MODEL = "text-embedding-3-large"
CHAT_MODEL = "gpt-4o-mini"

# Path Configuration
DOCUMENTS_PATH = "data/documents"
IMAGES_PATH = "data/images"

# Poppler and Tesseract paths (Windows)
POPPLER_PATH = os.getenv("POPPLER_PATH", r"C:\Release-24.08.0-0\poppler-24.08.0\Library\bin")
TESSERACT_PATH = os.getenv("TESSERACT_PATH", r"C:\Program Files\Tesseract-OCR")

# Add to PATH if not already there
if POPPLER_PATH and POPPLER_PATH not in os.environ["PATH"]:
    os.environ["PATH"] += os.pathsep + POPPLER_PATH
if TESSERACT_PATH and TESSERACT_PATH not in os.environ["PATH"]:
    os.environ["PATH"] += os.pathsep + TESSERACT_PATH