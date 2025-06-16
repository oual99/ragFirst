"""Streamlit conversational bot application for RAG system."""
import streamlit as st
import os
from pathlib import Path
import sys
import re
import traceback
from datetime import datetime
import json

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from src.document_processor import DocumentProcessor
from src.embeddings import EmbeddingGenerator
from src.database import WeaviateDatabase
from src.search import SearchEngine
from src.conversational_rag_engine import ConversationalRAGEngine
from src.conversation_manager import ConversationManager


# Page configuration
st.set_page_config(
    page_title="RAG Conversational Bot - BTP",
    page_icon="💬",
    layout="wide"
)


# Custom CSS for chat interface with better contrast
st.markdown("""
<style>
    /* Chat message styling with better contrast */
    .stChatMessage {
        border-radius: 10px;
        padding: 10px;
        margin: 5px 0;
    }
    
    /* Light mode specific styles */
    @media (prefers-color-scheme: light) {
        .stChatMessage[data-testid="user-message"] {
            background-color: #e3f2fd !important;
            color: #0d47a1 !important;
        }
        
        .stChatMessage[data-testid="assistant-message"] {
            background-color: #f5f5f5 !important;
            color: #212121 !important;
        }
        
        .source-box {
            background-color: #e8eaf6 !important;
            color: #1a237e !important;
            padding: 10px;
            border-radius: 5px;
            margin: 5px 0;
            font-size: 0.9em;
            border: 1px solid #c5cae9;
        }
        
        /* Ensure text in the main area is readable */
        .main .block-container {
            color: #212121 !important;
        }
        
        /* Fix sidebar text */
        .css-1d391kg, [data-testid="stSidebar"] {
            color: #212121 !important;
        }
        
        /* Fix input text */
        .stTextInput input {
            color: #212121 !important;
        }
    }
    
    /* Dark mode specific styles */
    @media (prefers-color-scheme: dark) {
        .stChatMessage[data-testid="user-message"] {
            background-color: #1e3a5f !important;
            color: #e3f2fd !important;
        }
        
        .stChatMessage[data-testid="assistant-message"] {
            background-color: #2e2e2e !important;
            color: #e0e0e0 !important;
        }
        
        .source-box {
            background-color: #3f51b5 !important;
            color: #e8eaf6 !important;
            padding: 10px;
            border-radius: 5px;
            margin: 5px 0;
            font-size: 0.9em;
            border: 1px solid #5c6bc0;
        }
    }
    
    /* General improvements for both modes */
    .stMarkdown {
        color: inherit !important;
    }
    
    /* Better button visibility */
    .stButton > button {
        border: 1px solid currentColor;
    }
    
    /* Improve metric visibility */
    [data-testid="metric-container"] {
        background-color: rgba(128, 128, 128, 0.1);
        padding: 10px;
        border-radius: 5px;
    }
    
    /* Fix expander text */
    .streamlit-expanderHeader {
        color: inherit !important;
    }
    
    /* Ensure code blocks are readable */
    .stCodeBlock {
        background-color: rgba(128, 128, 128, 0.1) !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
if 'db' not in st.session_state:
    st.session_state.db = None
if 'embedding_gen' not in st.session_state:
    st.session_state.embedding_gen = None
if 'search_engine' not in st.session_state:
    st.session_state.search_engine = None
if 'rag_engine' not in st.session_state:
    st.session_state.rag_engine = None
if 'conversation_manager' not in st.session_state:
    st.session_state.conversation_manager = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'search_cache' not in st.session_state:
    st.session_state.search_cache = {}
if 'current_context' not in st.session_state:
    st.session_state.current_context = []
if 'conversation_id' not in st.session_state:
    st.session_state.conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S")


def check_configuration():
    """Check if all required configuration is present."""
    missing_configs = []
    
    # Check vector DB type
    if not config.VECTOR_DB_TYPE:
        missing_configs.append("VECTOR_DB_TYPE")
    
    # Check database-specific configs
    if config.VECTOR_DB_TYPE.lower() == "qdrant":
        if not config.QDRANT_URL:
            missing_configs.append("QDRANT_URL")
        if not config.QDRANT_API_KEY:
            missing_configs.append("QDRANT_API_KEY")
    elif config.VECTOR_DB_TYPE.lower() == "weaviate":
        if not config.WEAVIATE_URL:
            missing_configs.append("WEAVIATE_URL")
        if not config.WEAVIATE_API_KEY:
            missing_configs.append("WEAVIATE_API_KEY")
    
    # Always need OpenAI
    if not config.OPENAI_API_KEY:
        missing_configs.append("OPENAI_API_KEY")
    
    return missing_configs

def create_vector_database():
    """Factory function to create the appropriate vector database based on config."""
    if config.VECTOR_DB_TYPE.lower() == "qdrant":
        from src.qdrant_database import QdrantDatabase
        return QdrantDatabase(
            url=config.QDRANT_URL,
            api_key=config.QDRANT_API_KEY
        )
    elif config.VECTOR_DB_TYPE.lower() == "weaviate":
        from src.database import WeaviateDatabase
        return WeaviateDatabase(
            url=config.WEAVIATE_URL,
            api_key=config.WEAVIATE_API_KEY,
            openai_api_key=config.OPENAI_API_KEY
        )
    else:
        raise ValueError(f"Unknown vector database type: {config.VECTOR_DB_TYPE}")

def initialize_system():
    """Initialize all system components."""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Get the vector DB name for display
    db_display_name = config.VECTOR_DB_TYPE.capitalize()
    
    try:
        # Check configuration first
        status_text.text("Checking configuration...")
        missing = check_configuration()
        if missing:
            st.error(f"Missing configuration: {', '.join(missing)}")
            st.info("Please check your .env file and ensure all required values are set.")
            return False
        progress_bar.progress(20)
        
        # Initialize database
        status_text.text(f"Connecting to {db_display_name}...")
        try:
            st.session_state.db = create_vector_database()
            st.success(f"✓ Connected to {db_display_name}")
        except Exception as e:
            st.error(f"Failed to connect to {db_display_name}: {str(e)}")
            return False
        progress_bar.progress(40)
        
        # Initialize embedding generator
        status_text.text("Initializing embedding generator...")
        try:
            st.session_state.embedding_gen = EmbeddingGenerator()
            st.success("✓ Embedding generator initialized")
        except Exception as e:
            st.error(f"Failed to initialize embeddings: {str(e)}")
            return False
        progress_bar.progress(60)
        
        # Initialize search engine
        status_text.text("Initializing search engine...")
        st.session_state.search_engine = SearchEngine(
            st.session_state.db,  # Pass the whole database object, not just client
            st.session_state.embedding_gen
        )
        st.success("✓ Search engine initialized")
        progress_bar.progress(70)
        
        # Initialize conversational RAG engine
        status_text.text("Initializing conversational RAG engine...")
        st.session_state.rag_engine = ConversationalRAGEngine()
        st.success("✓ Conversational RAG engine initialized")
        progress_bar.progress(85)
        
        # Initialize conversation manager
        status_text.text("Initializing conversation manager...")
        st.session_state.conversation_manager = ConversationManager()
        st.success("✓ Conversation manager initialized")
        progress_bar.progress(100)
        
        status_text.text("System initialized successfully!")
        st.session_state.initialized = True
        
        # Add welcome message
        if not st.session_state.messages:
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"Bonjour! Je suis votre assistant conversationnel spécialisé dans les documents BTP. Je suis connecté à {db_display_name} pour stocker et rechercher vos documents. Comment puis-je vous aider aujourd'hui?",
                "timestamp": datetime.now().isoformat()
            })
        
        return True
        
    except Exception as e:
        st.error(f"Initialization failed: {str(e)}")
        st.code(traceback.format_exc())
        return False
    finally:
        progress_bar.empty()
        status_text.empty()


def should_search_documents(query: str, recent_context: list) -> bool:
    """Determine if we need to search documents for this query."""
    
    query_lower = query.lower().strip()
    
    # Handle very short queries
    if len(query_lower) < 3:
        return False
    
    # Direct document references
    document_patterns = [
        r'\b(document|fichier|page|pdf|dossier)\b',
        r'\b(section|chapitre|partie|paragraphe)\b'
    ]
    
    # Question patterns
    question_patterns = [
        r'^(quel|quelle|quels|quelles)\b',
        r'^(où|ou)\b.*\?',
        r'^(comment|combien|pourquoi|quand|qui|quoi)\b',
        r'\b(trouve|recherche|cherche|localise)\b',
        r'\b(montre|affiche|présente|donne)\b.*\b(moi|nous)\b'
    ]
    
    # No-search patterns
    no_search_patterns = [
        r'^(merci|ok|d\'accord|compris|parfait)',
        r'^(bonjour|salut|bonsoir|hello|hi)\b',
        r'\b(explique|précise|reformule|clarifie)\b',
        r'^(oui|non|si|peut-être)\b',
        r'^\?+$',  # Just question marks
    ]
    
    # Check no-search patterns first
    for pattern in no_search_patterns:
        if re.search(pattern, query_lower):
            return False
    
    # Check search patterns
    for pattern in document_patterns + question_patterns:
        if re.search(pattern, query_lower):
            return True
    
    # Context-aware decision
    if recent_context:
        # If it's a very short follow-up, likely doesn't need search
        if len(query_lower.split()) <= 3:
            # Check if it's a pronoun reference
            pronoun_patterns = [r'^(il|elle|ce|ça|cela|celui)', r'\b(le|la|les)\b']
            if any(re.search(p, query_lower) for p in pronoun_patterns):
                return False
    
    # Check if it's a complete question (ends with ?)
    if query.strip().endswith('?') and len(query_lower.split()) > 3:
        return True
    
    # Default: search for queries > 5 words, don't search for shorter ones
    return len(query_lower.split()) > 4


def get_cached_search_results(query: str, cache_duration_minutes: int = 5):
    """Get cached search results if available and recent."""
    cache_key = query.lower().strip()
    if cache_key in st.session_state.search_cache:
        cached_data = st.session_state.search_cache[cache_key]
        cached_time = datetime.fromisoformat(cached_data['timestamp'])
        if (datetime.now() - cached_time).seconds < cache_duration_minutes * 60:
            return cached_data['results']
    return None


def cache_search_results(query: str, results: list):
    """Cache search results with timestamp."""
    cache_key = query.lower().strip()
    st.session_state.search_cache[cache_key] = {
        'results': results,
        'timestamp': datetime.now().isoformat()
    }
    
    # Limit cache size
    if len(st.session_state.search_cache) > 20:
        # Remove oldest entries
        sorted_cache = sorted(st.session_state.search_cache.items(), 
                            key=lambda x: x[1]['timestamp'])
        st.session_state.search_cache = dict(sorted_cache[-20:])

def reset_vector_database():
    """Reset the vector database by deleting and recreating the collection."""
    try:
        # Check if collection exists and has data
        if st.session_state.db and st.session_state.db.collection_exists(config.COLLECTION_NAME):
            # Get collection stats to check if it's empty
            stats = st.session_state.db.get_collection_stats(config.COLLECTION_NAME)
            
            if stats["object_count"] == 0:
                return False, "La base de données est déjà vide. Aucune réinitialisation nécessaire."
            
            # Delete the collection using the interface method
            success = st.session_state.db.delete_collection(config.COLLECTION_NAME)
            
            if not success:
                return False, "Erreur lors de la suppression de la collection."
            
            # Recreate the collection
            st.session_state.db.create_collection(config.COLLECTION_NAME)
            
            # Clear all related states
            st.session_state.search_cache = {}
            st.session_state.current_context = []
            
            # Clear conversation but keep system initialized
            clear_conversation()
            
            return True, "Base de données vectorielle réinitialisée avec succès!"
        else:
            # Collection doesn't exist, create it
            if st.session_state.db:
                st.session_state.db.create_collection(config.COLLECTION_NAME)
                return False, "La collection n'existait pas. Une nouvelle collection a été créée."
            else:
                return False, "La base de données n'est pas initialisée."
    except Exception as e:
        return False, f"Erreur lors de la réinitialisation: {str(e)}"
    
def display_message(message):
    """Display a message in the chat interface with proper formatting."""
    with st.chat_message(message["role"]):
        # Show contradiction warning if applicable
        if message.get("has_contradictions", False) and message["role"] == "assistant":
            st.warning("⚠️ Réponse contenant des informations contradictoires")
        
        st.markdown(message["content"])
        
        # Display timestamp
        if "timestamp" in message:
            st.caption(f"🕐 {datetime.fromisoformat(message['timestamp']).strftime('%H:%M')}")


def export_conversation():
    """Export the current conversation."""
    conversation_data = {
        "conversation_id": st.session_state.conversation_id,
        "messages": st.session_state.messages,
        "export_date": datetime.now().isoformat()
    }
    
    return json.dumps(conversation_data, ensure_ascii=False, indent=2)


def clear_conversation():
    """Clear the current conversation."""
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Conversation réinitialisée. Comment puis-je vous aider?",
        "timestamp": datetime.now().isoformat()
    }]
    st.session_state.search_cache = {}
    st.session_state.current_context = []
    st.session_state.conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S")


def main():
    st.title("💬 Assistant Conversationnel RAG - Documents BTP")
    
    # Sidebar for configuration and document management
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Show configuration status
        missing_configs = check_configuration()
        if missing_configs:
            st.error("Configuration manquante!")
            for config_item in missing_configs:
                st.warning(f"❌ {config_item} non configuré")
        else:
            st.success("✓ Configuration chargée")
        
        # Initialize system button
        if not st.session_state.initialized:
            if st.button("🚀 Initialiser le système", type="primary", disabled=bool(missing_configs)):
                initialize_system()
        else:
            st.success("✓ Système initialisé")
            
            # Collection stats
            if st.session_state.db:
                try:
                    stats = st.session_state.db.get_collection_stats(config.COLLECTION_NAME)
                    if stats["exists"]:
                        st.metric("Documents indexés", stats['object_count'])
                except Exception:
                    pass
        
        st.markdown("---")
        
        # Conversation management
        st.header("💬 Gestion de la conversation")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Effacer", help="Effacer la conversation actuelle"):
                clear_conversation()
                st.rerun()
        
        with col2:
            if st.button("💾 Exporter", help="Exporter la conversation"):
                conversation_json = export_conversation()
                st.download_button(
                    label="📥 Télécharger",
                    data=conversation_json,
                    file_name=f"conversation_{st.session_state.conversation_id}.json",
                    mime="application/json"
                )
        
        st.markdown("---")
        
        # Vector Database Management
        st.header("🗄️ Gestion de la base vectorielle")
        
        # Get current database status
        db_status = "Non initialisée"
        object_count = 0
        if st.session_state.db and st.session_state.db.collection_exists(config.COLLECTION_NAME):
            stats = st.session_state.db.get_collection_stats(config.COLLECTION_NAME)
            object_count = stats.get("object_count", 0)
            db_status = f"{object_count} documents indexés"
        
        st.info(f"État actuel: {db_status}")
        
        # Only show reset option if database has data
        if object_count > 0:
            # Add confirmation checkbox for safety
            confirm_reset = st.checkbox("Je confirme vouloir réinitialiser", key="confirm_reset")
            
            # Reset button
            if st.button("🔄 Réinitialiser la base vectorielle", 
                         type="secondary", 
                         disabled=not confirm_reset,
                         help="Efface toutes les données vectorielles"):
                
                # Show warning
                with st.spinner("Réinitialisation en cours..."):
                    success, message = reset_vector_database()
                    
                if success:
                    st.success(message)
                    # Force page rerun instead of modifying session state
                    st.rerun()
                else:
                    st.warning(message)
            
            if confirm_reset:
                st.warning("⚠️ Cette action supprimera toutes les données indexées!")
        else:
            st.info("✓ La base de données est vide")
        
        # Cache statistics
        if st.session_state.search_cache:
            st.metric("Recherches en cache", len(st.session_state.search_cache))
        
        st.markdown("---")
        
        # Document upload section
        if st.session_state.initialized:
            st.header("📄 Télécharger des documents")
            
            # Multiple file uploader
            uploaded_files = st.file_uploader(
                "Choisir un ou plusieurs fichiers PDF", 
                type="pdf",
                accept_multiple_files=True,
                help="Vous pouvez sélectionner plusieurs fichiers PDF en maintenant Ctrl (ou Cmd sur Mac)"
            )
            
            if uploaded_files:
                # Show file count and names with sizes
                st.info(f"📎 {len(uploaded_files)} fichier(s) sélectionné(s)")
                total_size = 0
                for file in uploaded_files:
                    file_size_mb = file.size / (1024 * 1024)
                    st.text(f"• {file.name} ({file_size_mb:.1f} MB)")
                    total_size += file_size_mb
                st.text(f"📊 Taille totale: {total_size:.1f} MB")
                
                if st.button(f"📤 Traiter {len(uploaded_files)} document(s)", type="primary"):
                    os.makedirs(config.DOCUMENTS_PATH, exist_ok=True)
                    os.makedirs(config.IMAGES_PATH, exist_ok=True)
                    
                    # Create containers for progress tracking
                    main_progress = st.progress(0)
                    main_status = st.empty()
                    
                    # Create a container for individual file progress
                    progress_container = st.container()
                    
                    # Initialize progress trackers for each file
                    file_progress_bars = {}
                    file_status_texts = {}
                    file_containers = {}
                    
                    with progress_container:
                        st.markdown("### 📊 Progression détaillée")
                        for idx, file in enumerate(uploaded_files):
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                file_containers[idx] = st.container()
                                with file_containers[idx]:
                                    st.markdown(f"**{file.name}**")
                                    file_progress_bars[idx] = st.progress(0)
                                    file_status_texts[idx] = st.empty()
                            with col2:
                                # Placeholder for timing info
                                file_containers[f"{idx}_time"] = st.empty()
                    
                    # Track results
                    processed_files = []
                    failed_files = []
                    total_segments = 0
                    
                    try:
                        import time
                        start_time = time.time()
                        
                        # Process each file
                        for idx, uploaded_file in enumerate(uploaded_files):
                            file_start_time = time.time()
                            
                            # Update main progress
                            main_progress.progress(idx / len(uploaded_files))
                            main_status.text(f"📄 Traitement du fichier {idx + 1}/{len(uploaded_files)}: {uploaded_file.name}")
                            
                            try:
                                # Step 1: Save file (10%)
                                file_status_texts[idx].text("💾 Sauvegarde du fichier...")
                                file_progress_bars[idx].progress(0.1)
                                
                                save_path = os.path.join(config.DOCUMENTS_PATH, uploaded_file.name)
                                with open(save_path, "wb") as f:
                                    f.write(uploaded_file.getbuffer())
                                
                                # Step 2: Initialize NEW processor (20%)
                                file_status_texts[idx].text("🔧 Initialisation du processeur unifié...")
                                file_progress_bars[idx].progress(0.2)
                                
                                # Use the NEW enhanced document processor
                                from src.document_processor import DocumentProcessor
                                processor = DocumentProcessor(openai_api_key=config.OPENAI_API_KEY)
                                
                                # Step 3: Process PDF with NEW unified processor
                                def update_processing_progress(progress, message):
                                    # Map processing progress from 20% to 80%
                                    actual_progress = 0.2 + (progress * 0.6)
                                    file_progress_bars[idx].progress(actual_progress)
                                    file_status_texts[idx].text(f"📖 {message}")
                                
                                file_status_texts[idx].text("📖 Analyse du document...")
                                processed_doc = processor.process_pdf(save_path, progress_callback=update_processing_progress)
                                
                                # Step 4: Extract chunks with NEW chunking system
                                file_status_texts[idx].text("📝 Extraction des chunks optimisés...")
                                file_progress_bars[idx].progress(0.85)
                                
                                extracted_data = processor.extract_text_with_metadata(processed_doc, save_path)
                                
                                # Step 5: Ensure collection exists (90%)
                                file_status_texts[idx].text("🗄️ Vérification de la base de données...")
                                file_progress_bars[idx].progress(0.9)
                                if not st.session_state.db.collection_exists(config.COLLECTION_NAME):
                                    st.session_state.db.create_collection(config.COLLECTION_NAME)
                                
                                # Step 6: Ingest data with progress callback
                                file_status_texts[idx].text(f"💾 Indexation de {len(extracted_data)} chunks...")
                                
                                # Create a callback for ingestion progress
                                def update_ingestion_progress(progress, message):
                                    # Map ingestion progress from 90% to 100%
                                    actual_progress = 0.9 + (progress * 0.1)
                                    file_progress_bars[idx].progress(actual_progress)
                                    file_status_texts[idx].text(f"💾 {message}")
                                
                                # Use the progress-enabled ingestion if available
                                if hasattr(st.session_state.db, 'ingest_text_data_with_progress'):
                                    success = st.session_state.db.ingest_text_data_with_progress(
                                        config.COLLECTION_NAME,
                                        extracted_data,
                                        st.session_state.embedding_gen,
                                        progress_callback=update_ingestion_progress
                                    )
                                else:
                                    # Fallback to regular ingestion
                                    file_progress_bars[idx].progress(0.95)
                                    success = st.session_state.db.ingest_text_data(
                                        config.COLLECTION_NAME,
                                        extracted_data,
                                        st.session_state.embedding_gen
                                    )
                                    file_progress_bars[idx].progress(1.0)
                                
                                # Calculate processing time
                                file_end_time = time.time()
                                processing_time = file_end_time - file_start_time
                                
                                if success:
                                    file_progress_bars[idx].progress(1.0)
                                    
                                    # Get document summary from processed_doc
                                    doc_summary = processed_doc.get("summary", {})
                                    
                                    # Enhanced status text with page type information
                                    file_status_texts[idx].text(
                                        f"✅ Terminé - {len(extracted_data)} chunks, "
                                        f"{doc_summary.get('total_pages', 0)} pages "
                                        f"({doc_summary.get('scanned_pages', 0)} scannées, "
                                        f"{doc_summary.get('native_pages', 0)} natives)"
                                    )
                                    file_containers[f"{idx}_time"].success(f"⏱️ {processing_time:.1f}s")
                                    
                                    processed_files.append({
                                        "name": uploaded_file.name,
                                        "chunks": len(extracted_data),
                                        "pages": doc_summary.get('total_pages', 0),
                                        "scanned_pages": doc_summary.get('scanned_pages', 0),
                                        "native_pages": doc_summary.get('native_pages', 0),
                                        "images": doc_summary.get('total_images', 0),
                                        "tables": doc_summary.get('total_tables', 0),
                                        "time": processing_time
                                    })
                                    total_segments += len(extracted_data)
                                else:
                                    file_status_texts[idx].text("❌ Échec de l'indexation")
                                    file_containers[f"{idx}_time"].error(f"⏱️ {processing_time:.1f}s")
                                    failed_files.append({
                                        "name": uploaded_file.name,
                                        "error": "Échec de l'indexation"
                                    })
                                    
                            except Exception as e:
                                file_progress_bars[idx].progress(1.0)
                                file_status_texts[idx].text(f"❌ Erreur: {str(e)[:50]}...")
                                file_containers[f"{idx}_time"].error("Échec")
                                failed_files.append({
                                    "name": uploaded_file.name,
                                    "error": str(e)
                                })
                        
                        # Update final main progress
                        main_progress.progress(1.0)
                        total_time = time.time() - start_time
                        main_status.text(f"✅ Traitement terminé en {total_time:.1f} secondes!")
                        
                        # Show enhanced results summary
                        st.markdown("---")
                        st.markdown("### 📈 Résumé du traitement")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Documents traités", f"{len(processed_files)}/{len(uploaded_files)}")
                        with col2:
                            st.metric("Chunks indexés", total_segments)
                        with col3:
                            st.metric("Temps total", f"{total_time:.1f}s")
                        
                        if processed_files:
                            # Calculate totals
                            total_pages = sum(f['pages'] for f in processed_files)
                            total_scanned = sum(f['scanned_pages'] for f in processed_files)
                            total_native = sum(f['native_pages'] for f in processed_files)
                            total_images = sum(f.get('images', 0) for f in processed_files)
                            total_tables = sum(f.get('tables', 0) for f in processed_files)
                            
                            # Show detailed breakdown
                            st.info(
                                f"📊 **Analyse détaillée:**\n"
                                f"- Pages totales: {total_pages} ({total_scanned} scannées, {total_native} natives)\n"
                                f"- Images trouvées: {total_images}\n"
                                f"- Tableaux trouvés: {total_tables}"
                            )
                            
                            avg_time = sum(f['time'] for f in processed_files) / len(processed_files)
                            st.success(f"✓ Temps moyen par document: {avg_time:.1f}s")
                            
                            # Show per-document details in expander
                            with st.expander("📋 Détails par document", expanded=False):
                                for doc in processed_files:
                                    st.text(
                                        f"📄 {doc['name']}\n"
                                        f"   • {doc['chunks']} chunks créés\n"
                                        f"   • {doc['pages']} pages ({doc['scanned_pages']} scannées, {doc['native_pages']} natives)\n"
                                        f"   • {doc.get('images', 0)} images, {doc.get('tables', 0)} tableaux\n"
                                        f"   • Temps: {doc['time']:.1f}s"
                                    )
                        
                        if failed_files:
                            with st.expander("❌ Erreurs détaillées", expanded=True):
                                for file_info in failed_files:
                                    st.error(f"**{file_info['name']}**: {file_info['error']}")
                        
                    except Exception as e:
                        main_status.text("❌ Erreur générale lors du traitement")
                        st.error(f"Erreur: {str(e)}")
                        if st.button("Voir les détails de l'erreur"):
                            st.code(traceback.format_exc())
            
        
        # Conversation settings
        st.markdown("---")
        st.header("⚙️ Paramètres de conversation")

        search_limit = st.slider(
            "Nombre de résultats à rechercher", 
            min_value=3,
            max_value=20,
            value=10 if 'use_reranking' in st.session_state and st.session_state.use_reranking else 3,
            help="Nombre de résultats à extraire de la base de données"
        )

        use_reranking = st.checkbox(
            "🎯 Utiliser le reranking intelligent", 
            value=False,
            key="use_reranking",
            help="Active un tri intelligent des résultats par GPT-4 pour une meilleure précision. Recherche 10 résultats puis sélectionne les 3 meilleurs."
        )

        # Show dynamic explanation
        if use_reranking:
            st.info(
                f"🎯 Mode précis activé:\n"
                f"• Recherche étendue: {search_limit} résultats\n"
                f"• Sélection intelligente des 3 meilleurs\n"
                f"• Temps supplémentaire: ~2-3 secondes"
            )
            # Force search_limit to at least 10 when reranking
            if search_limit < 10:
                search_limit = 10
                st.warning("Limite augmentée à 10 pour le reranking")
        else:
            st.info(
                f"🚀 Mode rapide activé:\n"
                f"• Recherche directe: {search_limit} résultats\n"
                f"• Pas de reranking"
            )

        # Final number of results after reranking
        final_result_count = 3 if use_reranking else search_limit
        # auto_search = st.checkbox("Recherche automatique intelligente", value=True)
        # show_thinking = st.checkbox("Afficher le processus de réflexion", value=False)
        auto_search = True # Always enable auto search for this demo
        show_thinking = True # Always show thinking process for this demo
    
    # Main chat interface
    if not st.session_state.initialized:
        st.info("👈 Veuillez initialiser le système pour commencer.")
    else:
        # Display conversation history
        for message in st.session_state.messages:
            display_message(message)
        
        # Chat input
        if prompt := st.chat_input("Posez votre question sur les documents BTP..."):
            # Add user message
            user_message = {
                "role": "user",
                "content": prompt,
                "timestamp": datetime.now().isoformat()
            }
            st.session_state.messages.append(user_message)
            display_message(user_message)
            
            # Process the query
            with st.chat_message("assistant"):
                thinking_placeholder = st.empty()
                
                try:
                    # Check if we need to search
                    need_search = should_search_documents(prompt, st.session_state.current_context)
                    
                    if show_thinking:
                        thinking_placeholder.info(
                            f"🤔 Analyse de la question... {'Recherche nécessaire' if need_search else 'Utilisation du contexte existant'}"
                        )
                    
                    search_results = []
                    formatted_results = []
                    
                    if need_search and auto_search:
                        # Check cache first
                        cached_results = get_cached_search_results(prompt)
                        
                        if cached_results:
                            if show_thinking:
                                thinking_placeholder.info("📋 Utilisation des résultats en cache...")
                            search_results = cached_results
                        else:
                            if show_thinking:
                                thinking_placeholder.info("🔍 Recherche dans les documents...")
                            
                            # Perform search - get more results if reranking is enabled
                            initial_limit = search_limit  # This will be 10+ if reranking is on
                            
                            search_results = st.session_state.search_engine.search_multimodal(
                                prompt, 
                                config.COLLECTION_NAME, 
                                limit=initial_limit
                            )
                            
                            # Apply reranking if enabled and we have results
                            if use_reranking and search_results and len(search_results) > 3:
                                if show_thinking:
                                    thinking_placeholder.info(
                                        f"🎯 Sélection intelligente des résultats... "
                                        f"({len(search_results)} → 3 meilleurs)"
                                    )
                                
                                # Rerank results
                                search_results = st.session_state.search_engine.rerank_results(
                                    query=prompt,
                                    search_results=search_results,
                                    top_k=3
                                )
                                
                                if show_thinking:
                                    thinking_placeholder.info(
                                        f"✅ Reranking terminé - "
                                        f"{len(search_results)} résultats sélectionnés"
                                    )
                            
                            # Cache results (whether reranked or not)
                            if search_results:
                                cache_search_results(prompt, search_results)
                        
                        # Rest of the search processing continues as before...
                        if search_results:
                            formatted_results = st.session_state.search_engine.format_search_results(
                                search_results
                            )
                            st.session_state.current_context = formatted_results
                    
                    # Get conversation history
                    conversation_history = st.session_state.conversation_manager.get_formatted_history(
                        st.session_state.messages[-10:]  # Last 10 messages
                    )
                    
                    if show_thinking:
                        thinking_placeholder.info("💭 Génération de la réponse...")
                    
                    # Generate response
                    response_data = st.session_state.rag_engine.generate_conversational_response(
                        query=prompt,
                        search_results=formatted_results if need_search else st.session_state.current_context,
                        conversation_history=conversation_history,
                        include_sources=need_search
                    )
                    
                    # Clear thinking placeholder
                    thinking_placeholder.empty()
                    
                    # Check for contradictions and display warning if found
                    if response_data.get("has_contradictions", False):
                        st.warning("⚠️ **Informations contradictoires détectées** - Veuillez vérifier les sources citées ci-dessous.")
                    
                    # Display response
                    st.markdown(response_data["response"])
                    if need_search and search_results:
                        # Add a small indicator of search mode used
                        mode_indicator = "🎯 Mode précis" if use_reranking else "🚀 Mode rapide"
                        st.caption(f"{mode_indicator} - {len(search_results)} sources utilisées")
                    
                    # Add assistant message to history
                    assistant_message = {
                        "role": "assistant",
                        "content": response_data["response"],
                        "timestamp": datetime.now().isoformat(),
                        "has_contradictions": response_data.get("has_contradictions", False)
                    }
                    
                    if need_search and response_data.get("sources"):
                        assistant_message["sources"] = response_data["sources"]
                    
                    st.session_state.messages.append(assistant_message)
                    
                    # # Display sources if available
                    # if need_search and response_data.get("sources"):
                    #     with st.expander("📚 Sources consultées", expanded=False):
                    #         for i, source in enumerate(response_data["sources"]):
                    #             st.markdown(f"""
                    #             <div class="source-box">
                    #             <b>Source {i+1}</b> - Distance: {source['distance']:.3f}<br>
                    #             📄 Document: {os.path.basename(source['document'])}<br>
                    #             📍 Page {source['page']} | Paragraphe {source['paragraph']}
                    #             </div>
                    #             """, unsafe_allow_html=True)
                    
                    # Suggest follow-up questions
                    if response_data.get("follow_up_suggestions"):
                        st.markdown("**💡 Questions suggérées:**")
                        cols = st.columns(len(response_data["follow_up_suggestions"]))
                        for i, suggestion in enumerate(response_data["follow_up_suggestions"]):
                            with cols[i]:
                                if st.button(suggestion, key=f"suggestion_{i}"):
                                    st.session_state.prompt_input = suggestion
                                    st.rerun()
                    
                except Exception as e:
                    thinking_placeholder.empty()
                    st.error(f"Erreur lors du traitement: {str(e)}")
                    if show_thinking:
                        with st.expander("Détails de l'erreur"):
                            st.code(traceback.format_exc())
        
        # Display helpful tips
        # with st.expander("💡 Conseils d'utilisation"):
        #     st.markdown("""
        #     - Posez des questions naturelles sur vos documents BTP
        #     - Je peux me souvenir du contexte de notre conversation
        #     - Demandez des clarifications ou des détails supplémentaires
        #     - Je peux faire des comparaisons entre différentes sections
        #     - N'hésitez pas à me demander de reformuler ou d'expliquer différemment
        #     """)


if __name__ == "__main__":
    os.makedirs("data/documents", exist_ok=True)
    os.makedirs("data/images", exist_ok=True)
    os.makedirs("data/conversations", exist_ok=True)
    
    main()