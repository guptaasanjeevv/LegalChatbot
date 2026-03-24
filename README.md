Legal Chatbot (RAG + Ollama + Streamlit)

An AI-powered **Legal Chatbot** that leverages **Retrieval-Augmented Generation (RAG)** with **local LLMs (Ollama - Llama3)** to answer legal queries based on uploaded PDF documents.

---

## Features

* Upload and process legal PDFs
* Hybrid search (Semantic + Keyword-based)
* Context-aware responses using RAG
* Clause comparison capability
* Dockerized deployment with Ollama
* Fast UI using Streamlit

---

## Architecture Overview
<img width="765" height="726" alt="image" src="https://github.com/user-attachments/assets/92481937-d0ac-417b-aa19-24a59ca7e876" />


## Project Structure

```
.
├── app.py                # Streamlit UI & API integration
├── LegalChatbot.py      # Core RAG logic (Vector DB + BM25 + Agent)
├── requirements.txt     # Python dependencies
├── Dockerfile           # Container setup
├── docker-compose.yml   # Multi-container setup (App + Ollama)
├── .dockerignore
└── pdfs/                # Folder for uploaded PDFs
```

---

## Tech Stack
Frontend/UI: Streamlit
LLM: Ollama (Llama3)
Embeddings: Sentence Transformers (all-MiniLM-L6-v2)
Vector DB: FAISS
Keyword Search: BM25
PDF Parsing: PyPDF

## How It Works
# 1. Document Ingestion
a. PDFs are loaded and parsed
b. Text is chunked into smaller segments

# 2. Hybrid Retrieval
a. Semantic search using FAISS
b. Keyword search using BM25
c. Combined results for better accuracy

# 3. LLM Processing
a. Context passed to Llama3 via Ollama
b. Generates grounded legal responses

# 4. Agent Logic
a. Detects "compare" queries
b. Performs clause comparison automatically

## Installation & Setup
# Docker Setup:
# 1. Build & Run
     docker-compose up --build

# 2. Access App
     Streamlit UI: http://localhost:8501
     Ollama API runs internally via Docker network

     
