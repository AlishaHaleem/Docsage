import streamlit as st
import openai
import os
from PyPDF2 import PdfReader
import docx
import json
from typing import List, Dict
import textwrap

# Set up OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

# UI Configuration
st.set_page_config(page_title="DocIQ Analyzer", layout="wide", page_icon="📚")

# Custom CSS for styling
st.markdown("""
<style>
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    .stButton>button {
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        padding: 8px 16px;
    }
    .stTextInput>div>div>input {
        border-radius: 8px;
    }
    .chat-message {
        padding: 12px 16px;
        border-radius: 8px;
        margin: 8px 0;
        max-width: 80%;
    }
    .user-message {
        background-color: #f0f2f6;
        margin-left: auto;
    }
    .assistant-message {
        background-color: #e3f2fd;
    }
    .chunk-card {
        padding: 12px;
        margin: 8px 0;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
    }
    .menu-item {
        padding: 12px;
        margin: 8px 0;
        border-radius: 8px;
        cursor: pointer;
    }
    .menu-item:hover {
        background-color: #f0f2f6;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chunks" not in st.session_state:
    st.session_state.chunks = {}
if "quiz" not in st.session_state:
    st.session_state.quiz = None
if "active_menu" not in st.session_state:
    st.session_state.active_menu = "1"

# --- Helper Functions ---
def chunk_text(text: str, chunk_size: int = 1000) -> List[str]:
    return textwrap.wrap(text, chunk_size)

def extract_text(file):
    if file.type == "application/pdf":
        reader = PdfReader(file)
        return "\n".join([page.extract_text() for page in reader.pages])
    elif file.type == "text/plain":
        return str(file.read(), "utf-8")
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(file)
        return "\n".join([para.text for para in doc.paragraphs])
    return ""

def generate_summary(chunk: str) -> str:
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Create a concise 2-sentence summary."},
            {"role": "user", "content": chunk}
        ],
        temperature=0.3
    )
    return response.choices[0].message.content

def generate_quiz(context: str) -> Dict:
    prompt = f"""Generate 3 MCQs from this content:
    {context}
    
    Return JSON format with: question, 4 options, correct_answer (0-3), explanation"""
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a quiz generation expert."},
            {"role": "user", "content": prompt}
        ],
        response_format={ "type": "json_object" },
        temperature=0.5
    )
    return json.loads(response.choices[0].message.content)

# --- Sidebar Menu ---
with st.sidebar:
    st.markdown("# How can I help you today?")
    st.markdown("Select an option from the menu below:")
    
    menu_options = {
        "1": "📚 Document Processing",
        "2": "❓ Q&A Chat",
        "3": "🧠 Quiz Generator",
        "4": "⚙️ Settings",
        "5": "🆕 New Chat"
    }
    
    for key, value in menu_options.items():
        if st.button(f"**{key}**\n### {value}", key=f"menu_{key}", 
                    use_container_width=True,
                    help=f"Select {value}"):
            st.session_state.active_menu = key
            if key == "5":
                st.session_state.messages = []
                st.session_state.chunks = {}
                st.session_state.quiz = None
                st.rerun()

# --- Main Content Area ---
if st.session_state.active_menu == "1":  # Document Processing
    st.header("📚 Document Processing")
    st.markdown("Upload documents to analyze and extract information")
    
    uploaded_files = st.file_uploader(
        "Choose files", 
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
    
    if st.button("Process Documents") and uploaded_files:
        with st.spinner("Processing..."):
            st.session_state.chunks = {}
            for file in uploaded_files:
                text = extract_text(file)
                chunks = chunk_text(text)
                st.session_state.chunks[file.name] = {
                    "full_text": text,
                    "chunks": chunks,
                    "summaries": [None] * len(chunks),
                    "flags": [False] * len(chunks)
                }
            st.success(f"Processed {len(uploaded_files)} documents")
    
    if st.session_state.chunks:
        st.subheader("📝 Document Chunks")
        for doc_name, doc_data in st.session_state.chunks.items():
            with st.expander(f"📄 {doc_name}"):
                for i, chunk in enumerate(doc_data["chunks"]):
                    with st.container():
                        st.markdown(f"**Chunk {i+1}**")
                        st.markdown(f'<div class="chunk-card">{chunk[:200]}...</div>', 
                                   unsafe_allow_html=True)
                        
                        col1, col2, col3 = st.columns([1,1,1])
                        with col1:
                            if st.button("📄 View", key=f"view_{doc_name}_{i}"):
                                st.session_state.current_chunk = {
                                    "doc": doc_name,
                                    "index": i,
                                    "text": chunk
                                }
                        with col2:
                            if st.button("📝 Summary", key=f"summary_{doc_name}_{i}"):
                                with st.spinner("Generating..."):
                                    st.session_state.chunks[doc_name]["summaries"][i] = generate_summary(chunk)
                        with col3:
                            st.session_state.chunks[doc_name]["flags"][i] = st.checkbox(
                                "🚩 Flag", 
                                key=f"flag_{doc_name}_{i}",
                                value=st.session_state.chunks[doc_name]["flags"][i]
                            )
                        
                        if doc_data["summaries"][i]:
                            st.info(f"**Summary:** {doc_data['summaries'][i]}")

elif st.session_state.active_menu == "2":  # Q&A Chat
    st.header("❓ Document Q&A")
    st.markdown("Ask questions about your uploaded documents")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.container():
            st.markdown(f'<div class="chat-message {"user-message" if message["role"] == "user" else "assistant-message"}">'
                        f'<b>{"You" if message["role"] == "user" else "Assistant"}</b><br>'
                        f'{message["content"]}</div>', 
                        unsafe_allow_html=True)
    
    # Question input
    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Generate response
        with st.spinner("Analyzing..."):
            context = "\n\n".join([doc["full_text"] for doc in st.session_state.chunks.values()]) if st.session_state.chunks else ""
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": f"Answer based on: {context}" if context else "Be helpful."},
                    *[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
                ]
            )
            answer = response.choices[0].message.content
            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.rerun()

elif st.session_state.active_menu == "3":  # Quiz Generator
    st.header("🧠 Quiz Generator")
    st.markdown("Create quizzes from your documents")
    
    if st.session_state.chunks:
        if st.button("Generate Quiz"):
            with st.spinner("Creating quiz..."):
                context = "\n\n".join([doc["full_text"] for doc in st.session_state.chunks.values()])
                st.session_state.quiz = generate_quiz(context)
        
        if st.session_state.quiz:
            st.subheader("Generated Quiz")
            for i, question in enumerate(st.session_state.quiz["questions"]):
                with st.container():
                    st.markdown(f"**Question {i+1}:** {question['question']}")
                    
                    selected = st.radio(
                        "Select an answer:",
                        question["options"],
                        key=f"quiz_{i}",
                        index=None,
                        label_visibility="collapsed"
                    )
                    
                    if selected:
                        if question["options"].index(selected) == question["correct_answer"]:
                            st.success(f"✅ Correct! {question['explanation']}")
                        else:
                            st.error(f"❌ Incorrect. {question['explanation']}")
    else:
        st.warning("Please upload and process documents first")

elif st.session_state.active_menu == "4":  # Settings
    st.header("⚙️ Settings")
    st.markdown("Configure your preferences")
    
    # Settings options would go here

# Footer
st.markdown("---")
st.markdown("**DocIQ Analyzer** can make mistakes. Consider verifying important information.")
