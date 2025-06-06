import os
import uuid
import pickle
import numpy as np
from typing import List, Dict
from fastapi import FastAPI, UploadFile, HTTPException, BackgroundTasks
from pydantic import BaseModel
from groq import Groq
from langchain_cohere import CohereEmbeddings # Updated import for CohereEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- CONFIG ────────────────────────────────────────────────────────────────────

# You must set these in Render's env‐vars: COHERE_API_KEY and GROQ_API_KEY
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not COHERE_API_KEY:
    logger.error("COHERE_API_KEY environment variable is required")
    raise RuntimeError("COHERE_API_KEY environment variable is required")
if not GROQ_API_KEY:
    logger.error("GROQ_API_KEY environment variable is required")
    raise RuntimeError("GROQ_API_KEY environment variable is required")

try:
    # Cohere embedder (no local model load; calls Cohere’s API)
    embeddings = CohereEmbeddings(model="embed-english-v2.0", cohere_api_key=COHERE_API_KEY) # Recommended newer model version
    logger.info("CohereEmbeddings initialized successfully.")
except Exception as e:
    logger.error(f"Error initializing CohereEmbeddings: {e}")
    raise RuntimeError(f"Failed to initialize CohereEmbeddings: {e}")

try:
    # Groq client
    groq_client = Groq(api_key=GROQ_API_KEY)
    logger.info("Groq client initialized successfully.")
except Exception as e:
    logger.error(f"Error initializing Groq client: {e}")
    raise RuntimeError(f"Failed to initialize Groq client: {e}")

# Chunking parameters
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# Limits
MAX_DOCUMENTS = 20        # max docs you’ll accept
MAX_CHUNKS_PER_DOC = 100  # assume ~100 chunks per document. This is an upper bound.

# Vector store file (persist embeddings + texts + metadata)
VECTOR_STORE_PATH = "vectors.pkl" # This file will be on Render's ephemeral disk.

# --- IN‐MEMORY VECTOR STORE STRUCTURE ──────────────────────────────────────────

# We store a dict:
# {
#   "ids":    List[str],
#   "texts":  List[str],
#   "embs":   List[List[float]],  # dense vectors (Cohere dims)
#   "source": List[str],          # doc_id for each chunk
# }

# Initialize vector_store globally. Load existing data if available.
# This assumes the 'vectors.pkl' file is small enough to fit into Render's RAM.
# For larger datasets, a persistent database is recommended.
vector_store: Dict = {"ids": [], "texts": [], "embs": [], "source": []}
if os.path.exists(VECTOR_STORE_PATH):
    try:
        with open(VECTOR_STORE_PATH, "rb") as f:
            loaded_store = pickle.load(f)
            # Basic validation to ensure the loaded data has the expected structure
            if all(key in loaded_store for key in ["ids", "texts", "embs", "source"]):
                vector_store = loaded_store
                logger.info(f"Loaded existing vector store from {VECTOR_STORE_PATH} with {len(vector_store['ids'])} chunks.")
            else:
                logger.warning(f"Invalid format found in {VECTOR_STORE_PATH}. Starting with an empty store.")
    except Exception as e:
        logger.error(f"Error loading vector store from {VECTOR_STORE_PATH}: {e}. Starting with an empty store.")
else:
    logger.info("No existing vector store found. Starting with an empty store.")

def persist_store():
    """Write vector_store back to disk."""
    try:
        with open(VECTOR_STORE_PATH, "wb") as f:
            pickle.dump(vector_store, f)
        logger.info(f"Vector store persisted to {VECTOR_STORE_PATH}.")
    except Exception as e:
        logger.error(f"Error persisting vector store to {VECTOR_STORE_PATH}: {e}")

# --- FASTAPI APP & Pydantic MODELS ─────────────────────────────────────────────

app = FastAPI()

class QueryRequest(BaseModel):
    question: str

# --- DOCUMENT PROCESSING ──────────────────────────────────────────────────────

def process_document_text(text: str, doc_id: str):
    """
    1. Chunk `text` into ~CHUNK_SIZE tokens (roughly) with overlap.
    2. Embed each chunk via Cohere API.
    3. Append to in‐memory store and persist.
    """
    try:
        # 1) split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
        )
        chunks = text_splitter.split_text(text)
        logger.info(f"Document {doc_id} split into {len(chunks)} chunks.")

        # Enforce a max‐chunks cap
        if len(chunks) > MAX_CHUNKS_PER_DOC:
            logger.warning(f"Document {doc_id} has {len(chunks)} chunks, capping at {MAX_CHUNKS_PER_DOC}.")
            chunks = chunks[:MAX_CHUNKS_PER_DOC]

        # 2) get embeddings from Cohere
        #    CohereEmbeddings.embed_documents returns List[List[float]]
        embs: List[List[float]] = embeddings.embed_documents(chunks)
        logger.info(f"Generated embeddings for {len(embs)} chunks of document {doc_id}.")

        # 3) acquire lock before modifying shared vector_store (important for concurrency)
        # Note: For simplicity and given Python's GIL, a global lock isn't strictly
        # necessary for simple list appends, but it's good practice for shared state.
        # However, for a single-process FastAPI background task, current approach is fine.
        # If using multiple worker processes, a more robust locking mechanism or
        # a process-safe data structure would be needed.
        for i, chunk_text in enumerate(chunks):
            chunk_id = f"{doc_id}_{i}"
            vector_store["ids"].append(chunk_id)
            vector_store["texts"].append(chunk_text)
            vector_store["embs"].append(embs[i])
            vector_store["source"].append(doc_id)

        # 4) persist to disk (after all appends for this doc are done)
        persist_store()
        logger.info(f"Finished processing and persisting document {doc_id}.")
    except Exception as e:
        logger.error(f"Error processing document {doc_id}: {e}")
        # Consider a mechanism to report failed background tasks to the user if critical

@app.post("/upload")
async def upload_document(file: UploadFile, background_tasks: BackgroundTasks):
    """
    Client uploads a text file (plain .txt or .md). We read it entirely as text.
    Then we enqueue `process_document_text` via BackgroundTasks so that FastAPI responds immediately.
    """
    # 1) enforce max documents: roughly check if existing doc_ids >= MAX_DOCUMENTS
    # Using a set for unique doc_ids is good.
    existing_doc_ids = set(vector_store["source"])
    if len(existing_doc_ids) >= MAX_DOCUMENTS:
        logger.warning(f"Document limit reached ({len(existing_doc_ids)} >= {MAX_DOCUMENTS}). Rejecting upload.")
        raise HTTPException(status_code=400, detail=f"Document limit reached. Max {MAX_DOCUMENTS} documents allowed.")

    # 2) read file bytes
    if file.content_type not in ["text/plain", "text/markdown"]:
        raise HTTPException(status_code=400, detail="Unsupported file type. Only plain text (.txt) or markdown (.md) allowed.")

    try:
        content_bytes = await file.read()
        text_str = content_bytes.decode("utf-8")
        logger.info(f"Received upload: {file.filename}, size: {len(content_bytes)} bytes.")
    except UnicodeDecodeError:
        logger.error(f"Unable to decode upload {file.filename} as UTF-8 text.")
        raise HTTPException(status_code=400, detail="Unable to decode upload as UTF-8 text")
    except Exception as e:
        logger.error(f"Error reading uploaded file {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Error reading uploaded file: {e}")

    # 3) new doc_id and background‐enqueue
    doc_id = str(uuid.uuid4())
    background_tasks.add_task(process_document_text, text_str, doc_id)
    logger.info(f"Document {doc_id} enqueued for processing.")

    return {"id": doc_id, "status": "processing", "message": "Document is being processed in the background."}

# --- SIMILARITY SEARCH & QUERY ─────────────────────────────────────────────────

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two 1D NumPy arrays."""
    # Add a small epsilon to avoid division by zero for zero vectors, though unlikely with embeddings
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0 # Or raise an error, depending on desired behavior for zero vectors
    return float(np.dot(a, b) / (norm_a * norm_b))

def retrieve_top_k(question: str, k: int = 3):
    """
    1) Embed the `question` via CohereEmbedding API (list of floats).
    2) Compute cosine similarity against all stored embeddings.
    3) Return top‐k chunks (dicts) with highest similarity.
    """
    if not vector_store["embs"]:
        logger.info("No embeddings in store, returning empty for retrieve_top_k.")
        return []

    try:
        # 1) embed question
        q_emb: List[float] = embeddings.embed_query(question)
        q_vec = np.array(q_emb).reshape((1, -1)) # Ensure q_vec is 2D for dot product
        logger.debug("Question embedded.")

        # Ensure embs_array is numpy array for efficient operations
        embs_array = np.array(vector_store["embs"])

        # Handle case where embs_array might be empty after conversion or has wrong shape
        if embs_array.size == 0 or embs_array.shape[1] != q_vec.shape[1]:
            logger.warning("Embeddings array is empty or has mismatched dimensions for similarity calculation.")
            return []

        # 2) compute similarities using optimized numpy operations
        # Cosine similarity formula: A · B / (||A|| * ||B||)
        # We calculate (A · B) and (||A|| * ||B||) separately.

        dot_products = np.dot(embs_array, q_vec.T).flatten()  # shape (N,)
        embs_norms = np.linalg.norm(embs_array, axis=1)      # shape (N,)
        q_norm = np.linalg.norm(q_vec)                       # scalar

        # Avoid division by zero for any zero-norm vectors
        denominator = embs_norms * q_norm
        # Create a mask for non-zero denominators
        non_zero_denominators_mask = denominator != 0
        cos_sims = np.zeros_like(dot_products, dtype=float) # Initialize with zeros
        cos_sims[non_zero_denominators_mask] = dot_products[non_zero_denominators_mask] / denominator[non_zero_denominators_mask]

        logger.debug("Cosine similarities calculated.")

        # 3) pick top k indices
        # Use np.argsort for sorting and then slicing for top k
        top_k_idx = np.argsort(cos_sims)[::-1][:k] # Sort descending and take top k
        logger.debug(f"Retrieved top {k} indices.")

        # 4) build result list of dicts
        results = []
        for idx in top_k_idx:
            # Ensure index is within bounds before accessing
            if 0 <= idx < len(vector_store["ids"]):
                results.append({
                    "chunk_id": vector_store["ids"][idx],
                    "text": vector_store["texts"][idx],
                    "source": vector_store["source"][idx],
                    "score": float(cos_sims[idx]) # Convert numpy float to Python float
                })
        logger.info(f"Retrieved {len(results)} top chunks for question.")
        return results
    except Exception as e:
        logger.error(f"Error in retrieve_top_k: {e}")
        return []


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
        logger.info("No relevant chunks found for the query.")
        return {"answer": "No relevant information found in the indexed documents. Please upload more documents."}

    # 2) build context
    context = "\n\n".join([chunk["text"] for chunk in top_chunks])
    logger.debug(f"Context built from {len(top_chunks)} chunks.")

    # 3) call Groq API
    try:
        response = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": f"You are a helpful assistant. Answer the question using only the provided context. If the answer cannot be found in the context, state that clearly."},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {request.question}"}
            ],
            model="llama3-8b-8192", # Or "mixtral-8x7b-32768" for potentially better results but higher latency
            max_tokens=512,
            temperature=0.3,
            stream=False # For a single response, stream=False is fine.
        )
        answer = response.choices[0].message.content
        logger.info("Groq API call successful.")
    except Exception as e:
        logger.error(f"Groq API error during query: {e}")
        raise HTTPException(status_code=500, detail=f"Groq API error: {str(e)}")

    return {"answer": answer}

# --- HEALTH CHECK ──────────────────────────────────────────────────────────────

@app.get("/healthz")
def health_check():
    indexed_docs_count = len(set(vector_store["source"]))
    total_chunks_count = len(vector_store["ids"])
    logger.info(f"Health check: {indexed_docs_count} indexed documents, {total_chunks_count} total chunks.")
    return {
        "status": "ok",
        "indexed_documents": indexed_docs_count,
        "total_chunks_indexed": total_chunks_count,
        "vector_store_size_bytes": os.path.getsize(VECTOR_STORE_PATH) if os.path.exists(VECTOR_STORE_PATH) else 0
    }
    
