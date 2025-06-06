import os
import uuid
import numpy as np
from typing import List, Dict, Union
from fastapi import FastAPI, UploadFile, HTTPException, BackgroundTasks
from pydantic import BaseModel
from groq import Groq
from langchain_cohere import CohereEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Standard Library Imports (these should always be at the very top) ---
import logging # Import logging module first
import tempfile
import shutil

# --- CONFIGURE LOGGING HERE (BEFORE ANY USE OF 'logger') ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # Initialize the logger object
# -----------------------------------------------------------

# --- CORRECTED IMPORTS FOR DOCUMENT LOADERS ---
from langchain_community.document_loaders import TextLoader, PyPDFLoader # Check if these are still directly available
try:
    from langchain_community.document_loaders import Docx2textLoader # Try top-level again, in case it's re-exported
except ImportError as e:
    # Use the now-defined 'logger' object for error reporting
    logger.warning(f"Could not import Docx2textLoader from top-level: {e}. Trying alternate path.")
    try:
        from langchain_community.document_loaders.word_document import Docx2textLoader # Alternate path
        logger.info("Successfully imported Docx2textLoader from langchain_community.document_loaders.word_document.")
    except ImportError as e_alt:
        logger.error(f"Could not import Docx2textLoader from any known path. Ensure python-docx is installed and langchain-community is compatible. Original error: {e_alt}")
        raise ImportError("Failed to import Docx2textLoader. Check dependencies and LangChain version.") from e_alt

# ---------------------------------------------
from langchain_chroma import Chroma
from langchain_core.documents import Document as LangchainDocument # Alias to avoid conflict with Pydantic BaseModel


# --- REST OF YOUR CODE REMAINS THE SAME ---

# --- CONFIG ────────────────────────────────────────────────────────────────────

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not COHERE_API_KEY:
    logger.error("COHERE_API_KEY environment variable is required")
    raise RuntimeError("COHERE_API_KEY environment variable is required")
if not GROQ_API_KEY:
    logger.error("GROQ_API_KEY environment variable is required")
    raise RuntimeError("GROQ_API_KEY environment variable is required")

try:
    embeddings = CohereEmbeddings(model="embed-english-v3.0", cohere_api_key=COHERE_API_KEY)
    logger.info("CohereEmbeddings initialized successfully.")
except Exception as e:
    logger.error(f"Error initializing CohereEmbeddings: {e}")
    raise RuntimeError(f"Failed to initialize CohereEmbeddings: {e}")

try:
    groq_client = Groq(api_key=GROQ_API_KEY)
    logger.info("Groq client initialized successfully.")
except Exception as e:
    logger.error(f"Error initializing Groq client: {e}")
    raise RuntimeError(f"Failed to initialize Groq client: {e}")

# Chunking parameters
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# Limits
MAX_DOCUMENTS = 20
MAX_CHUNKS_PER_DOC = 100
MAX_DOCUMENT_FILE_SIZE_MB = 10
MAX_DOCUMENT_TEXT_LENGTH = 2 * 1024 * 1024 # Approx 2MB of text content

# --- VECTOR DATABASE CONFIG ───────────────────────────────────────────────────

CHROMA_DB_DIR = "./chroma_db"
os.makedirs(CHROMA_DB_DIR, exist_ok=True)
logger.info(f"ChromaDB persistence directory: {CHROMA_DB_DIR}")

try:
    vector_store = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
    logger.info("ChromaDB initialized successfully.")
except Exception as e:
    logger.error(f"Error initializing ChromaDB: {e}")
    raise RuntimeError(f"Failed to initialize ChromaDB: {e}")

# --- FASTAPI APP & Pydantic MODELS ─────────────────────────────────────────────

app = FastAPI(
    title="RAG API",
    description="Retrieval-Augmented Generation API for document querying.",
    version="1.0.0",
)

class QueryRequest(BaseModel):
    question: str

# --- DOCUMENT PROCESSING ──────────────────────────────────────────────────────

FILE_LOADERS = {
    "application/pdf": PyPDFLoader,
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": Docx2textLoader,
    "text/plain": TextLoader,
    "text/markdown": TextLoader,
}

def load_document_from_temp_file(file_path: str, content_type: str) -> List[LangchainDocument]:
    """
    Loads a document from a temporary file path using the appropriate LangChain loader.
    """
    loader_class = None
    for mime_type, cls in FILE_LOADERS.items():
        if content_type.startswith(mime_type):
            loader_class = cls
            break

    if not loader_class:
        raise ValueError(f"Unsupported content type: {content_type}")

    loader = loader_class(file_path)
    return loader.load()


def process_document(doc_id: str, file_path: str, original_filename: str, content_type: str):
    """
    1. Load document content from file based on type.
    2. Chunk text into manageable sizes.
    3. Add chunks to ChromaDB.
    """
    try:
        raw_documents: List[LangchainDocument] = load_document_from_temp_file(file_path, content_type)
        if not raw_documents:
            logger.warning(f"No content loaded for document {doc_id} from {original_filename}.")
            return

        full_text_content = "\n".join([doc.page_content for doc in raw_documents])

        if len(full_text_content) > MAX_DOCUMENT_TEXT_LENGTH:
            logger.warning(f"Document {original_filename} (ID: {doc_id}) text length "
                           f"({len(full_text_content)} chars) exceeds MAX_DOCUMENT_TEXT_LENGTH "
                           f"({MAX_DOCUMENT_TEXT_LENGTH} chars). Truncating.")
            full_text_content = full_text_content[:MAX_DOCUMENT_TEXT_LENGTH]

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
        )
        chunks = text_splitter.split_text(full_text_content)

        if len(chunks) > MAX_CHUNKS_PER_DOC:
            logger.warning(f"Document {doc_id} has {len(chunks)} chunks, capping at {MAX_CHUNKS_PER_DOC}.")
            chunks = chunks[:MAX_CHUNKS_PER_DOC]

        if not chunks:
            logger.warning(f"No chunks generated for document {doc_id} after splitting. Skipping.")
            return

        metadatas = []
        for i in range(len(chunks)):
            metadatas.append({
                "doc_id": doc_id,
                "filename": original_filename,
                "chunk_index": i,
                "source_type": content_type
            })

        vector_store.add_texts(
            texts=chunks,
            metadatas=metadatas,
            ids=[f"{doc_id}_{i}" for i in range(len(chunks))]
        )
        logger.info(f"Successfully processed and added {len(chunks)} chunks for document {original_filename} (ID: {doc_id}) to ChromaDB.")

    except ValueError as ve:
        logger.error(f"Error loading document {original_filename} (ID: {doc_id}): {ve}")
    except Exception as e:
        logger.error(f"Error processing document {original_filename} (ID: {doc_id}): {e}")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.debug(f"Cleaned up temporary file: {file_path}")


@app.post("/upload")
async def upload_document(file: UploadFile, background_tasks: BackgroundTasks):
    """
    Client uploads a document file (plain text, markdown, PDF, DOCX).
    We save it to a temporary file, then enqueue `process_document` via BackgroundTasks.
    """
    # Count unique documents in ChromaDB
    unique_doc_ids = set()
    try:
        all_chroma_metadatas = vector_store.get(include=['metadatas'])['metadatas']
        for metadata in all_chroma_metadatas:
            if 'doc_id' in metadata:
                unique_doc_ids.add(metadata['doc_id'])
    except Exception as e:
        logger.warning(f"Could not retrieve existing document IDs from ChromaDB for limit check: {e}")
        pass

    if len(unique_doc_ids) >= MAX_DOCUMENTS:
        logger.warning(f"Document limit reached ({len(unique_doc_ids)} >= {MAX_DOCUMENTS}). Rejecting upload.")
        raise HTTPException(status_code=400, detail=f"Document limit reached. Max {MAX_DOCUMENTS} unique documents allowed.")

    if file.content_type not in FILE_LOADERS:
        supported_types = ", ".join(FILE_LOADERS.keys())
        logger.warning(f"Unsupported file type: {file.content_type}. Supported: {supported_types}")
        raise HTTPException(status_code=400, detail=f"Unsupported file type. Supported types: {supported_types}")

    file_size_mb = file.size / (1024 * 1024) if file.size else 0
    if file_size_mb > MAX_DOCUMENT_FILE_SIZE_MB:
        logger.warning(f"File {file.filename} size ({file_size_mb:.2f}MB) exceeds limit ({MAX_DOCUMENT_FILE_SIZE_MB}MB).")
        raise HTTPException(status_code=400, detail=f"File size exceeds {MAX_DOCUMENT_FILE_SIZE_MB} limit.")

    doc_id = str(uuid.uuid4())
    temp_file_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            temp_file_path = temp_file.name
            file_content = await file.read()
            temp_file.write(file_content)
        logger.info(f"Saved uploaded file {file.filename} to temporary path: {temp_file_path}")

        background_tasks.add_task(process_document, doc_id, temp_file_path, file.filename, file.content_type)

    except Exception as e:
        logger.error(f"Error during file upload or temporary save for {file.filename}: {e}")
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise HTTPException(status_code=500, detail=f"Failed to process upload: {e}")

    return {"id": doc_id, "status": "processing", "message": "Document is being processed in the background."}

# --- SIMILARITY SEARCH & QUERY ─────────────────────────────────────────────────

@app.post("/query")
async def query_documents(request: QueryRequest):
    """
    1) Retrieve top-k chunks for the question using ChromaDB.
    2) Build a prompt: “Answer using context: < concatenated chunks >”
    3) Call Groq’s chat endpoint for generation.
    4) Return the generated answer.
    """
    if vector_store._collection.count() == 0:
        logger.info("ChromaDB is empty, no documents to query.")
        return {"answer": "No documents indexed yet. Please upload documents first."}

    try:
        retrieved_docs: List[LangchainDocument] = vector_store.similarity_search(
            query=request.question,
            k=3
        )
        logger.info(f"ChromaDB retrieved {len(retrieved_docs)} relevant chunks for the query.")

        if not retrieved_docs:
            logger.info("No relevant chunks found by ChromaDB for the query.")
            return {"answer": "No relevant information found in the indexed documents."}

        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        logger.debug(f"Context built from {len(retrieved_docs)} chunks.")

        response = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": f"You are a helpful assistant. Answer the question using only the provided context. If the answer cannot be found in the context, state that clearly and do not make up information."},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {request.question}"}
            ],
            model="llama3-8000b-8192",
            max_tokens=512,
            temperature=0.3,
            stream=False
        )
        answer = response.choices[0].message.content
        logger.info("Groq API call successful.")
    except Exception as e:
        logger.error(f"Error during query processing or Groq API call: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during query: {str(e)}")

    return {"answer": answer}

# --- METADATA & HEALTH CHECKS ──────────────────────────────────────────────────

@app.get("/")
async def root():
    """
    Root endpoint to confirm the RAG API is running.
    """
    return {"message": "RAG API is running!", "status": "ok", "api_version": app.version}

@app.get("/healthz")
def health_check():
    """
    Basic health check including indexed document count.
    """
    try:
        total_chunks_indexed = vector_store._collection.count()
        
        unique_doc_ids = set()
        all_chroma_metadatas = vector_store.get(include=['metadatas'])['metadatas']
        for metadata in all_chroma_metadatas:
            if 'doc_id' in metadata:
                unique_doc_ids.add(metadata['doc_id'])
        indexed_docs_count = len(unique_doc_ids)

        chroma_db_size_bytes = 0
        if os.path.exists(CHROMA_DB_DIR) and os.path.isdir(CHROMA_DB_DIR):
            for dirpath, dirnames, filenames in os.walk(CHROMA_DB_DIR):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    if os.path.isfile(fp):
                        chroma_db_size_bytes += os.path.getsize(fp)

        logger.info(f"Health check: {indexed_docs_count} indexed documents, {total_chunks_indexed} total chunks in ChromaDB. DB size: {chroma_db_size_bytes} bytes.")
        return {
            "status": "ok",
            "indexed_documents": indexed_docs_count,
            "total_chunks_indexed": total_chunks_indexed,
            "chroma_db_size_bytes": chroma_db_size_bytes
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@app.get("/documents")
async def list_documents_metadata():
    """
    Endpoint to view metadata of all processed documents.
    """
    try:
        all_metadatas = vector_store.get(include=['metadatas'])['metadatas']
        
        documents_info = {}
        for meta in all_metadatas:
            doc_id = meta.get('doc_id')
            if doc_id:
                if doc_id not in documents_info:
                    documents_info[doc_id] = {
                        "doc_id": doc_id,
                        "filename": meta.get('filename', 'N/A'),
                        "source_type": meta.get('source_type', 'N/A'),
                        "chunk_count": 0,
                        "chunks": []
                    }
                documents_info[doc_id]["chunk_count"] += 1
        
        return {"documents": list(documents_info.values()), "total_unique_documents": len(documents_info)}
    except Exception as e:
        logger.error(f"Error listing document metadata: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve document metadata: {str(e)}")
        
