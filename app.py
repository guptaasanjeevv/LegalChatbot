import os
import requests
from LegalChatbot import agent_router, PDF_FOLDER, KeywordStore, VectorStore, load_pdfs, chunk_text
# from ollama import ollama_llm
import streamlit as st

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

def ollama_llm(prompt):
    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": "llama3",
            "prompt": prompt,
            "stream": False
        }
    )
    return response.json()["response"]

# -------------------------------
# CONFIG
# -------------------------------
st.set_page_config(page_title="Legal Chatbot", layout="wide")
# -------------------------------

# -------------------------------
# BUILD INDEX
# -------------------------------
@st.cache_resource
def build_index():
    docs = load_pdfs(PDF_FOLDER)
    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunk_text(doc))

    vector_store = VectorStore()
    vector_store.add(all_chunks)

    keyword_store = KeywordStore(all_chunks)

    return vector_store, keyword_store

vector_store, keyword_store = build_index()

# STREAMLIT UI
st.title("Legal Chatbot (RAG + Ollama)")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")
if uploaded_file:
    with open(os.path.join(PDF_FOLDER, uploaded_file.name), "wb") as f:
        f.write(uploaded_file.read())
    st.success("PDF uploaded! Please refresh to re-index.")

query = st.text_input("Ask a legal question:")

if st.button("Ask") and query:
    with st.spinner("Thinking..."):
        response = agent_router(query)
    st.markdown(f"**Answer:** {response}")