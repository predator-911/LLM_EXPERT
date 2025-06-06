# main.py

# main.py

import os
import uuid
import pickle
import numpy as np
from typing import List, Dict
from fastapi import FastAPI, UploadFile, HTTPException, BackgroundTasks
from pydantic import BaseModel
# Corrected import
from groq import Groq  # ✅ Updated to use Groq class
from langchain.embeddings import CohereEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ─── CONFIG ────────────────────────────────────────────────────────────────────

# You must set these in Render's env‐vars: COHERE_API_KEY and GROQ_API_KEY
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not COHERE_API_KEY:
    raise RuntimeError("COHERE_API_KEY environment variable is required")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY environment variable is required")

# Cohere embedder (no local model load; calls Cohere’s API)
embeddings = CohereEmbeddings(model="embed-english-v2", cohere_api_key=COHERE_API_KEY)

# Groq client - Corrected initialization
groq_client = Groq(api_key=GROQ_API_KEY)  # ✅ Using Groq class

# ... rest of the code remains unchanged ...
# You must set these in Render's env‐vars: COHERE_API_KEY and GROQ_API_KEY
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not COHERE_API_KEY:
    raise RuntimeError("COHERE_API_KEY environment variable is required")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY environment variable is required")

# Cohere embedder (no local model load; calls Cohere’s API)
embeddings = CohereEmbeddings(model="embed-english-v2", cohere_api_key=COHERE_API_KEY)

# Groq client
groq_client = GroqClient(api_key=GROQ_API_KEY)

# Chunking parameters
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# Limits
MAX_DOCUMENTS = 20        # max docs you’ll accept
MAX_CHUNKS_PER_DOC = 100  # assume ~100 chunks per document

# Vector store file (persist embeddings + texts + metadata)
VECTOR_STORE_PATH = "vectors.pkl"

# ─── IN‐MEMORY VECTOR STORE STRUCTURE ──────────────────────────────────────────

# We store a dict:
# {
#   "ids":    List[str],
#   "texts":  List[str],
#   "embs":   List[List[float]],  # dense vectors (Cohere dims)
#   "source": List[str],          # doc_id for each chunk
# }
if os.path.exists(VECTOR_STORE_PATH):
    with open(VECTOR_STORE_PATH, "rb") as f:
        vector_store: Dict = pickle.load(f)
else:
    vector_store = {"ids": [], "texts": [], "embs": [], "source": []}


def persist_store():
    """Write vector_store back to disk."""
    with open(VECTOR_STORE_PATH, "wb") as f:
        pickle.dump(vector_store, f)


# ─── FASTAPI APP & Pydantic MODELS ─────────────────────────────────────────────

app = FastAPI()


class QueryRequest(BaseModel):
    question: str


# ─── DOCUMENT PROCESSING ──────────────────────────────────────────────────────

def process_document_text(text: str, doc_id: str):
    """
    1. Chunk `text` into ~CHUNK_SIZE tokens (roughly) with overlap.
    2. Embed each chunk via Cohere API.
    3. Append to in‐memory store and persist.
    """
    # 1) split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)

    # Enforce a max‐chunks cap
    if len(chunks) > MAX_CHUNKS_PER_DOC:
        chunks = chunks[:MAX_CHUNKS_PER_DOC]

    # 2) get embeddings from Cohere
    #    CohereEmbeddings.embed_documents returns List[List[float]]
    embs: List[List[float]] = embeddings.embed_documents(chunks)

    # 3) append to our store
    for i, chunk_text in enumerate(chunks):
        chunk_id = f"{doc_id}_{i}"
        vector_store["ids"].append(chunk_id)
        vector_store["texts"].append(chunk_text)
        vector_store["embs"].append(embs[i])
        vector_store["source"].append(doc_id)

    # 4) persist to disk
    persist_store()


@app.post("/upload")
async def upload_document(file: UploadFile, background_tasks: BackgroundTasks):
    """
    Client uploads a text file (plain .txt or .md). We read it entirely as text.
    Then we enqueue `process_document_text` via BackgroundTasks so that FastAPI responds immediately.
    """
    # 1) enforce max documents: roughly check if existing doc_ids >= MAX_DOCUMENTS
    existing_doc_ids = set(vector_store["source"])
    if len(existing_doc_ids) >= MAX_DOCUMENTS:
        raise HTTPException(400, "Document limit reached")

    # 2) read file bytes
    content_bytes = await file.read()
    try:
        text_str = content_bytes.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(400, "Unable to decode upload as UTF-8 text")

    # 3) new doc_id and background‐enqueue
    doc_id = str(uuid.uuid4())
    background_tasks.add_task(process_document_text, text_str, doc_id)

    return {"id": doc_id, "status": "processing"}


# ─── SIMILARITY SEARCH & QUERY ─────────────────────────────────────────────────

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two 1D NumPy arrays."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def retrieve_top_k(question: str, k: int = 3):
    """
    1) Embed the `question` via CohereEmbedding API (list of floats).
    2) Compute cosine similarity against all stored embeddings.
    3) Return top‐k chunks (dicts) with highest similarity.
    """
    if not vector_store["embs"]:
        return []

    # 1) embed question
    q_emb: List[float] = embeddings.embed_query(question)  # length = dim of cohere embed

    # 2) compute similarities
    embs_array = np.array(vector_store["embs"])  # shape (N, D)
    q_vec = np.array(q_emb).reshape((1, -1))      # shape (1, D)

    # cosine similarity: (embs_array ⋅ q_vec) / (||embs|| * ||q_vec||)
    # We'll do a batched dot then norms:
    dot_products = embs_array @ q_vec.T          # shape (N, 1)
    embs_norms = np.linalg.norm(embs_array, axis=1).reshape((-1, 1))  # shape (N,1)
    q_norm = np.linalg.norm(q_vec)               # scalar
    cos_sims = (dot_products / (embs_norms * q_norm)).flatten()  # shape (N,)

    # 3) pick top k indices
    top_k_idx = np.argsort(-cos_sims)[:k]

    # 4) build result list of dicts
    results = []
    for idx in top_k_idx:
        results.append({
            "chunk_id": vector_store["ids"][idx],
            "text": vector_store["texts"][idx],
            "source": vector_store["source"][idx],
            "score": float(cos_sims[idx])
        })
    return results


@app.post("/query")
async def query_documents(request: QueryRequest):
    """
    1) Retrieve top-k chunks for the question.
    2) Build a prompt: “Answer using context: < concatenated chunks >”
    3) Call Groq’s chat endpoint for generation.
    4) Return the generated answer.
    """
    # 1) similarity search
    top_chunks = retrieve_top_k(request.question, k=3)

    if not top_chunks:
        return {"answer": "No documents indexed yet."}

    # 2) build context
    context = "\n\n".join([chunk["text"] for chunk in top_chunks])

    # 3) call Groq API
    try:
        response = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": f"Answer using context:\n{context}"},
                {"role": "user", "content": request.question}
            ],
            model="llama3-8b-8192",
            max_tokens=512,
            temperature=0.3
        )
        answer = response.choices[0].message.content
    except Exception as e:
        raise HTTPException(500, f"Groq API error: {str(e)}")

    return {"answer": answer}


# ─── HEALTH CHECK ──────────────────────────────────────────────────────────────

@app.get("/healthz")
def health_check():
    return {"status": "ok", "indexed_docs": len(set(vector_store["source"]))}
