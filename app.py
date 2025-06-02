"""Streamlit conversational bot application for RAG system."""
import streamlit as st
import os
from pathlib import Path
import sys
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
    
    if not config.WEAVIATE_URL:
        missing_configs.append("WEAVIATE_URL")
    if not config.WEAVIATE_API_KEY:
        missing_configs.append("WEAVIATE_API_KEY")
    if not config.OPENAI_API_KEY:
        missing_configs.append("OPENAI_API_KEY")
    
    return missing_configs


def initialize_system():
    """Initialize all system components."""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
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
        status_text.text("Connecting to Weaviate...")
        try:
            st.session_state.db = WeaviateDatabase()
            st.success("✓ Connected to Weaviate")
        except Exception as e:
            st.error(f"Failed to connect to Weaviate: {str(e)}")
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
            st.session_state.db.client,
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
                "content": "Bonjour! Je suis votre assistant conversationnel spécialisé dans les documents BTP. Comment puis-je vous aider aujourd'hui?",
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
    # Keywords that typically require new searches
    search_keywords = ['recherche', 'trouve', 'montre', 'quel', 'où', 'combien', 
                      'document', 'page', 'information sur', 'détails sur']
    
    # Keywords that typically don't require searches
    no_search_keywords = ['merci', 'ok', 'd\'accord', 'compris', 'claire', 
                         'explique', 'précise', 'reformule', 'répète']
    
    query_lower = query.lower()
    
    # Check if it's a follow-up question
    if any(keyword in query_lower for keyword in no_search_keywords):
        return False
    
    # Check if it explicitly asks for document search
    if any(keyword in query_lower for keyword in search_keywords):
        return True
    
    # If we have recent context and the query seems related, don't search
    if recent_context and len(query_lower.split()) < 10:
        return False
    
    # Default to searching for new topics
    return True


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
            
            # Delete the collection
            st.session_state.db.client.collections.delete(config.COLLECTION_NAME)
            
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
            st.session_state.db.create_collection(config.COLLECTION_NAME)
            return False, "La collection n'existait pas. Une nouvelle collection a été créée."
    except Exception as e:
        return False, f"Erreur lors de la réinitialisation: {str(e)}"

def display_message(message):
    """Display a message in the chat interface with proper formatting."""
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Display sources if available
        # if "sources" in message and message["sources"]:
        #     with st.expander("📚 Sources consultées", expanded=False):
        #         for i, source in enumerate(message["sources"]):
        #             st.markdown(f"""
        #             <div class="source-box">
        #             <b>Source {i+1}</b> - Distance: {source['distance']:.3f}<br>
        #             📄 Document: {os.path.basename(source['document'])}<br>
        #             📍 Page {source['page']} | Paragraphe {source['paragraph']}
        #             </div>
        #             """, unsafe_allow_html=True)
        
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
            uploaded_file = st.file_uploader("Choisir un fichier PDF", type="pdf")
            
            if uploaded_file is not None:
                if st.button("📤 Traiter le document"):
                    os.makedirs(config.DOCUMENTS_PATH, exist_ok=True)
                    os.makedirs(config.IMAGES_PATH, exist_ok=True)
                    
                    save_path = os.path.join(config.DOCUMENTS_PATH, uploaded_file.name)
                    
                    with open(save_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    


                    # Create progress bar and status containers
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    try:
                        # Step 1: Initialize processor
                        status_text.text("📄 Initialisation du processeur de documents...")
                        progress_bar.progress(10)
                        processor = DocumentProcessor(images_output_dir=config.IMAGES_PATH)
                        
                        # Step 2: Process PDF
                        status_text.text("📖 Extraction du contenu du PDF...")
                        progress_bar.progress(25)
                        raw_data = processor.process_pdf(save_path)
                        
                        # Step 3: Extract text with metadata
                        status_text.text("📝 Extraction du texte et des métadonnées...")
                        progress_bar.progress(40)
                        extracted_data = processor.extract_text_with_metadata(raw_data, save_path)
                        
                        # Step 4: Check/Create collection
                        status_text.text("🗄️ Vérification de la collection...")
                        progress_bar.progress(50)
                        if not st.session_state.db.collection_exists(config.COLLECTION_NAME):
                            st.session_state.db.create_collection(config.COLLECTION_NAME)
                        
                        # Step 5: Ingest data with progress
                        status_text.text(f"💾 Indexation de {len(extracted_data)} segments...")
                        progress_bar.progress(60)
                        
                        # For detailed progress during ingestion, we need to modify the ingest_text_data method
                        # For now, we'll show the ingestion as a single step
                        success = st.session_state.db.ingest_text_data(
                            config.COLLECTION_NAME,
                            extracted_data,
                            st.session_state.embedding_gen
                        )
                        
                        if success:
                            progress_bar.progress(100)
                            status_text.text("✅ Traitement terminé!")
                            st.success(f"✓ {len(extracted_data)} segments traités avec succès!")
                            # st.balloons()
                        else:
                            status_text.text("❌ Échec de l'indexation")
                            st.error("Erreur lors de l'indexation des données")
                            
                    except Exception as e:
                        status_text.text("❌ Erreur lors du traitement")
                        st.error(f"Erreur: {str(e)}")
                    finally:
                        # Clean up progress indicators after a short delay
                        import time
                        time.sleep(1)
                        progress_bar.empty()
                        status_text.empty()
        
        # Conversation settings
        st.markdown("---")
        st.header("⚙️ Paramètres de conversation")
        
        search_limit = st.slider("Nombre de résultats", 1, 10, 3)
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
                            
                            # Perform search
                            search_results = st.session_state.search_engine.search_multimodal(
                                prompt, 
                                config.COLLECTION_NAME, 
                                limit=search_limit
                            )
                            
                            # Cache results
                            if search_results:
                                cache_search_results(prompt, search_results)
                        
                        if search_results:
                            formatted_results = st.session_state.search_engine.format_search_results(
                                search_results
                            )
                            # Update current context
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
                    
                    # Display response
                    st.markdown(response_data["response"])
                    
                    # Add assistant message to history
                    assistant_message = {
                        "role": "assistant",
                        "content": response_data["response"],
                        "timestamp": datetime.now().isoformat()
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