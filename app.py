import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import os
import time
import uuid
import re
import PIL.Image
from google import genai
from pdf_parser import process_pdf

st.set_page_config(page_title="ARIL | Law Assistance", layout="wide", initial_sidebar_state="expanded")

# --- Custom CSS for Claude-like Layout & Navy/Gold Styling ---
custom_css = """
<style>
/* Typography & Base Theme */
* { font-family: 'Inter', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
h1, h2, h3, h4, h5, h6 { color: #C9A84C !important; font-family: 'Georgia', serif !important; }

/* Main layout padding overrides */
.block-container {
    padding-top: 1rem !important;
    padding-bottom: 6rem !important;
    max-width: 900px !important;
}

/* Hide Default Streamlit Header */
header[data-testid="stHeader"] { background-color: transparent !important; }

/* Sidebar - Chat History */
[data-testid="stSidebar"] {
    background-color: #0A1628 !important;
    border-right: 1px solid #1E2A3A !important;
}
.new-chat-btn button {
    width: 100%;
    background-color: transparent !important;
    color: #F5F5F0 !important;
    border: 1px solid #C9A84C !important;
    border-radius: 6px !important;
    padding: 10px !important;
    transition: all 0.3s ease !important;
    font-weight: 500 !important;
}
.new-chat-btn button:hover {
    background-color: rgba(201, 168, 76, 0.1) !important;
}
.chat-hist-btn button {
    width: 100%;
    text-align: left !important;
    justify-content: flex-start !important;
    background-color: transparent !important;
    color: #A0B0C0 !important;
    border: none !important;
    padding: 8px 12px !important;
    margin-bottom: 4px !important;
}
.chat-hist-btn button:hover {
    background-color: #1E2A3A !important;
    color: #F5F5F0 !important;
}
.chat-hist-btn.active button {
    background-color: #1E2A3A !important;
    color: #C9A84C !important;
    border-left: 3px solid #C9A84C !important;
}

/* Chat Messages Default reset */
[data-testid="stChatMessage"] {
    background-color: transparent !important;
    border: none !important;
    padding: 0 !important;
    margin-bottom: 24px !important;
}

/* User Message Bubble */
div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) {
    display: flex;
    flex-direction: row-reverse !important;
}
div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) [data-testid="chatAvatarIcon-user"] {
    display: none !important; /* Hide user avatar */
}
div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) [data-testid="stMarkdownContainer"] {
    background-color: #1E2A3A !important; /* Darker navy bubble */
    color: #F5F5F0 !important;
    padding: 12px 18px !important;
    border-radius: 18px 18px 4px 18px !important;
    max-width: 100%;
    margin-left: auto;
}

/* Assistant Message Bubble */
div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-assistant"]) [data-testid="chatAvatarIcon-assistant"] {
    background-color: #0A1628 !important;
    color: #C9A84C !important;
    border: 2px solid #C9A84C !important;
    display: flex;
    align-items: center;
    justify-content: center;
}
div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-assistant"]) [data-testid="stMarkdownContainer"] {
    color: #F5F5F0 !important;
    padding: 4px 0 !important;
}

/* Chat Input Bar */
[data-testid="stChatInput"] {
    background-color: #0D1F3C !important;
    border: 1px solid #8B7536 !important;
    border-radius: 12px !important;
    padding-left: 5px !important;
}
[data-testid="stChatInput"] textarea {
    color: #F5F5F0 !important;
}
/* Style the SVG icons in the chat input */
[data-testid="stChatInput"] button svg {
    fill: #C9A84C !important;
    color: #C9A84C !important;
}

/* Expander/Cards Theme override to match law card */
[data-testid="stExpander"] {
    background-color: #0D1F3C !important;
    border: 1px solid #C9A84C !important;
    border-radius: 8px !important;
    margin: 10px 0;
}
[data-testid="stExpander"] summary {
    border-bottom: 1px solid #1E2A3A !important;
    background-color: rgba(201, 168, 76, 0.05) !important;
}
[data-testid="stExpander"] summary p {
    color: #C9A84C !important;
    font-weight: 600;
}
[data-testid="stExpander"] summary:hover p {
    color: #F0C040 !important;
}

/* Settings Button Inside Sidebar */
.sidebar-settings [data-testid="stPopover"] {
    width: 100% !important;
    margin-bottom: 1rem !important;
    position: relative !important;
    top: auto !important;
    left: auto !important;
    right: auto !important;
}
.sidebar-settings [data-testid="stPopover"] > button {
    background: transparent !important;
    border: 1px solid #1E2A3A !important;
    color: #A0B0C0 !important;
    width: 100% !important;
    font-size: 1rem !important;
    padding: 8px !important;
    transition: all 0.3s ease !important;
    text-align: left !important;
    justify-content: flex-start !important;
}
.sidebar-settings [data-testid="stPopover"] > button:hover {
    color: #C9A84C !important;
    border-color: #C9A84C !important;
}

</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# --- State Management ---
if "conversations" not in st.session_state:
    st.session_state.conversations = {} # dict of chat_id -> {"title": "...", "timestamp": float, "messages": []}
if "current_chat_id" not in st.session_state:
    new_id = str(uuid.uuid4())
    st.session_state.current_chat_id = new_id
    st.session_state.conversations[new_id] = {
        "title": "New Conversation",
        "timestamp": time.time(),
        "messages": []
    }

if "api_key" not in st.session_state:
    st.session_state.api_key = "" # User needs to enter API key

chat_data = st.session_state.conversations[st.session_state.current_chat_id]

# --- Core Logic & Database ---
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(APP_DIR, "chroma_db")
client = chromadb.PersistentClient(path=DB_PATH)
collection_name = "law_sections"
collection = client.get_or_create_collection(name=collection_name)

def store_in_chroma(law_dict, source_name):
    if len(law_dict) == 0: return
    sections = list(law_dict.keys())
    texts = list(law_dict.values())
    embeddings = embedder.encode(texts).tolist()
    unique_ids = [f"{source_name}_{s}" for s in sections]
    collection.upsert(
        ids=unique_ids, embeddings=embeddings, documents=texts,
        metadatas=[{"section": s, "source": source_name} for s in sections]
    )

def process_and_index_pdfs():
    APP_DIR = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(APP_DIR, "dataset")
    if not os.path.exists(dataset_dir):
        return False, f"Dataset directory '{dataset_dir}' not found."
    pdf_files = [f for f in os.listdir(dataset_dir) if f.lower().endswith('.pdf')]
    if not pdf_files:
        return False, f"No PDF files found in '{dataset_dir}'."
    
    total_sections = 0
    for pdf_file in pdf_files:
        file_path = os.path.join(dataset_dir, pdf_file)
        with open(file_path, "rb") as f:
            pdf_bytes = f.read()
        law_dict = process_pdf(pdf_bytes)
        if law_dict:
            store_in_chroma(law_dict, pdf_file)
            total_sections += len(law_dict)
    
    if total_sections > 0:
        return True, f"Indexed {total_sections} sections from {len(pdf_files)} files."
    return False, "No sections could be parsed."

# --- Layout: Sidebar (Chat History) ---
with st.sidebar:
    # Settings Button at top of sidebar
    st.markdown("<div class='sidebar-settings'>", unsafe_allow_html=True)
    with st.popover("⚙️ Settings", help="Configuration and Database Management"):
        st.subheader("Settings")
        st.session_state.api_key = st.text_input("Gemini API Key", value=st.session_state.api_key, type="password")
        st.divider()
        st.write("**Database Management**")
        if st.button("Process & Index Local Law PDFs", use_container_width=True):
            with st.spinner("Processing..."):
                success, msg = process_and_index_pdfs()
                if success:
                    st.success(msg)
                else:
                    st.error(msg)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='new-chat-btn'>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([0.8, 0.2])
    with col1:
        if st.button("➕ New Chat", use_container_width=True):
            new_id = str(uuid.uuid4())
            st.session_state.conversations[new_id] = {
                "title": "New Conversation",
                "timestamp": time.time(),
                "messages": []
            }
            st.session_state.current_chat_id = new_id
            st.rerun()
    with col2:
        if st.button("🗑️", help="Clear all history"):
            st.session_state.show_clear_confirm = True

    st.markdown("</div>", unsafe_allow_html=True)
    
    if st.session_state.get("show_clear_confirm", False):
        st.warning("Delete all chat history?")
        c1, c2 = st.columns(2)
        if c1.button("Yes", use_container_width=True):
            st.session_state.conversations = {}
            new_id = str(uuid.uuid4())
            st.session_state.current_chat_id = new_id
            st.session_state.conversations[new_id] = {
                "title": "New Conversation",
                "timestamp": time.time(),
                "messages": []
            }
            st.session_state.show_clear_confirm = False
            st.rerun()
        if c2.button("No", use_container_width=True):
            st.session_state.show_clear_confirm = False
            st.rerun()

    st.divider()
    
    # Sort conversations by timestamp desc
    sorted_convos = sorted(st.session_state.conversations.items(), key=lambda x: x[1]['timestamp'], reverse=True)
    
    for chat_id, conv_data in sorted_convos:
        # Highlight active chat
        is_active = "active" if chat_id == st.session_state.current_chat_id else ""
        st.markdown(f"<div class='chat-hist-btn {is_active}'>", unsafe_allow_html=True)
        title_text = f"💬 {conv_data['title'][:25]}{'...' if len(conv_data['title'])>25 else ''}"
        if st.button(title_text, key=f"btn_{chat_id}", use_container_width=True):
            st.session_state.current_chat_id = chat_id
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

# --- Layout: Main Chat Interface ---
if not chat_data["messages"]:
    st.markdown("<h1 style='text-align: center; margin-top: 20vh; font-size: 3rem;'>ARIL</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #8B7536; font-size: 1.2rem;'>Your Law Assistance</p>", unsafe_allow_html=True)

# Render Message History
for msg in chat_data["messages"]:
    if msg["role"] == "user":
        # Custom HTML for User Message (Right Aligned, Navy Bubble)
        html = f"""
        <div style="display: flex; justify-content: flex-end; margin-bottom: 24px;">
            <div style="background-color: #1E2A3A; color: #F5F5F0; padding: 12px 18px; border-radius: 18px 18px 4px 18px; max-width: 80%; line-height: 1.5;">
                {msg['content']}
            </div>
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)
        if "images" in msg and msg["images"]:
            col1, col2 = st.columns([1, 0.01])
            with col1:
                st.markdown("<div style='display: flex; justify-content: flex-end;'>", unsafe_allow_html=True)
                for img in msg["images"]:
                    st.image(img, width=300)
                st.markdown("</div>", unsafe_allow_html=True)
    else:
        # Assistant Message
        with st.chat_message("assistant", avatar="⚖️"):
            if "content" in msg and msg["content"]:
                st.write(msg["content"])
            if "images" in msg and msg["images"]:
                for img in msg["images"]:
                    st.image(img, width=300)
            
            # Render the expandable gold-bordered law cards directly inside the flow
            if "law_cards" in msg and msg["law_cards"]:
                for card in msg["law_cards"]:
                    with st.expander(f"✨ Section {card['section']} — {card['source']}"):
                        st.write(card['text'])

# Bottom Fixed Chat Input (Text Only)
query = st.chat_input("Ask a legal question...")

if query:
    text_query = str(query).strip() if query else ""

    if text_query:
        # Append User Message
        user_msg_dict = {"role": "user", "content": text_query}
        chat_data["messages"].append(user_msg_dict)
        
        # Set chat title to the first user question
        if chat_data["title"] == "New Conversation" and text_query:
            chat_data["title"] = text_query
            
        chat_data["timestamp"] = time.time()
        st.rerun()

# --- Handle Assistant Processing On Rerun ---
if chat_data["messages"] and chat_data["messages"][-1]["role"] == "user":
    last_user_msg = chat_data["messages"][-1]
    
    # We must explicitly render the just-added user message first if we didn't just rerun from loop
    # Actually, the loop rendered it just before the input since st.rerun happened
    
    user_text = last_user_msg["content"]
    
    with st.chat_message("assistant", avatar="⚖️"):
        status_placeholder = st.empty()
        status_placeholder.markdown("*Analyzing and searching statutes...*")
        
        context_docs = []
        law_cards = []
        sections_to_search = []
        
        # 1. Exact Match Checking
        if user_text.strip().isdigit() or re.match(r"^\d+[A-Za-z]*$", user_text.strip()):
            sections_to_search.append(user_text.strip())
            
        # 2. AI Prompt to extract specific section
        api_key = st.session_state.api_key
        if api_key and not sections_to_search:
            try:
                client_genai = genai.Client(api_key=api_key)
                prompt_ext = f"Identify the most relevant Indian law section number(s) (e.g., IPC, BNS) for this query. Return ONLY the section numbers as a comma-separated list. If you don't know, return UNKNOWN.\nQuery: {user_text}"
                resp_ext = client_genai.models.generate_content(model='gemini-2.5-flash', contents=[prompt_ext])
                ext_text = resp_ext.text.strip()
                if ext_text and "UNKNOWN" not in ext_text.upper():
                    matches = re.findall(r'\b\d+[A-Za-z]*\b', ext_text)
                    if matches:
                        sections_to_search = matches
            except Exception as e:
                pass
                
        # 3. Exact Database Lookup
        if sections_to_search:
            for sec in sections_to_search:
                result = collection.get(where={"section": sec})
                if result and result['documents']:
                    for idx, doc in enumerate(result['documents']):
                        source = result['metadatas'][idx].get("source", "Unknown")
                        context_docs.append(f"Section {sec} ({source}): {doc}")
                        law_cards.append({"section": sec, "source": source, "text": doc})
                        
        # 4. Semantic Lookup
        if user_text:
            query_embedding = embedder.encode(user_text).tolist()
            results = collection.query(query_embeddings=[query_embedding], n_results=5)
            if results and results['documents'] and len(results['documents'][0]) > 0:
                for idx, doc in enumerate(results['documents'][0]):
                    section_num = results['metadatas'][0][idx]['section']
                    source = results['metadatas'][0][idx].get("source", "Unknown")
                    if not any(entry['section'] == section_num for entry in law_cards):
                        context_docs.append(f"Section {section_num} ({source}): {doc}")
                        law_cards.append({"section": section_num, "source": source, "text": doc})

        # Render the law cards in the chat flow instantly
        if law_cards:
            for card in law_cards:
                with st.expander(f"✨ Section {card['section']} — {card['source']}"):
                    st.write(card['text'])

        # 5. Build AI Response
        assistant_reply = ""
        if api_key and context_docs:
            try:
                client_g = genai.Client(api_key=api_key)
                prompt = f"""You are ARIL, an expert Legal AI Assistant. 
                
CRITICAL INSTRUCTIONS:
1. Provide localized legal advice based ONLY on the provided database laws. 
2. Formulate a final response. Do NOT reiterate that you've searched the database. Just cleanly answer.
3. If NONE of the retrieved sections apply accurately to the scenario, say: "I could not find a specific law in the database matching this scenario."
4. Format beautifully with markdown.

Retrieved Law Sections:
{chr(10).join(context_docs)}

User Query/Scenario: {user_text}"""

                contents = [prompt]
                    
                response = client_g.models.generate_content(model='gemini-2.5-flash', contents=contents)
                assistant_reply = response.text
                status_placeholder.write(assistant_reply)
            except Exception as e:
                assistant_reply = f"Error calling Gemini API: {e}"
                status_placeholder.error(assistant_reply)
        elif not context_docs:
            assistant_reply = "I couldn't find any relevant sections in the current database. Try processing more local PDFs or adjust your query."
            status_placeholder.write(assistant_reply)
        else:
            assistant_reply = "💡 Enter your Gemini API Key in Settings (Top Right Menu) to get AI-powered legal advice based on these law sections."
            status_placeholder.info(assistant_reply)

        # Append to state and rerender
        chat_data["messages"].append({
            "role": "assistant",
            "content": assistant_reply,
            "law_cards": law_cards
        })
        st.rerun()
