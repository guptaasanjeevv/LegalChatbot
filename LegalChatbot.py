import os

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
from rank_bm25 import BM25Okapi
import requests

PDF_FOLDER = "pdfs"
os.makedirs(PDF_FOLDER, exist_ok=True)

# -------------------------------
# LOAD EMBEDDING MODEL
# -------------------------------
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# -------------------------------
# PDF LOADER
# -------------------------------
def load_pdfs(folder):
    documents = []

    for file in os.listdir(folder):
        if file.endswith(".pdf"):
            reader = PdfReader(os.path.join(folder, file))
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""

            documents.append(text)

    return documents


# -------------------------------
# CHUNKING
# -------------------------------
def chunk_text(text, chunk_size=800, overlap=150):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks


# -------------------------------
# VECTOR STORE
# -------------------------------
class VectorStore:
    def __init__(self):
        self.texts = []
        self.embeddings = None
        self.index = None

    def add(self, chunks):
        self.texts.extend(chunks)
        embeddings = embedding_model.encode(chunks)

        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = list(self.embeddings) + list(embeddings)

        self.index = faiss.IndexFlatL2(len(embeddings[0]))
        self.index.add(self.embeddings)

    def search(self, query, k=5):
        query_vec = embedding_model.encode([query])
        distances, indices = self.index.search(query_vec, k)
        return [self.texts[i] for i in indices[0]]


# -------------------------------
# BM25 STORE
# -------------------------------
class KeywordStore:
    def __init__(self, chunks):
        self.tokenized = [chunk.split() for chunk in chunks]
        self.bm25 = BM25Okapi(self.tokenized)
        self.chunks = chunks

    def search(self, query, k=5):
        scores = self.bm25.get_scores(query.split())
        top_k = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self.chunks[i] for i in top_k]


# -------------------------------
# OLLAMA CALL
# -------------------------------
def ollama_llm(prompt):
    #url = "http://localhost:11434/api/generate"
    url = "http://host.docker.internal:11434/api/generate"
    payload = {
        "model": "llama3",
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(url, json=payload)
    return response.json()["response"]


# -------------------------------
# HYBRID SEARCH
# -------------------------------
def hybrid_search(query, vector_store, keyword_store, k=5):
    semantic = vector_store.search(query, k)
    keyword = keyword_store.search(query, k)

    return list(set(semantic + keyword))[:k]


# -------------------------------
# AGENT ROUTER
# -------------------------------
def agent_router(query):
    results = hybrid_search(query, vector_store, keyword_store)
    context = "\n\n".join(results)

    if "compare" in query.lower() and len(results) >= 2:
        prompt = f"""
Compare the following clauses:

Clause A:
{results[0]}

Clause B:
{results[1]}

Explain differences, risks, implications.
"""
        return ollama_llm(prompt)

    prompt = f"""
You are a legal assistant.

Context:
{context}

Question:
{query}

Answer clearly based only on context.
"""
    return ollama_llm(prompt)


# -------------------------------
# INGEST DATA ON STARTUP
# -------------------------------
documents = load_pdfs(PDF_FOLDER)

all_chunks = []
for doc in documents:
    all_chunks.extend(chunk_text(doc))

vector_store = VectorStore()
vector_store.add(all_chunks)

keyword_store = KeywordStore(all_chunks)

print(f"Loaded {len(all_chunks)} chunks")

