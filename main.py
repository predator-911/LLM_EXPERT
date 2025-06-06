# main.py
from fastapi import FastAPI, UploadFile, HTTPException, BackgroundTasks
from pydantic import BaseModel
import os
from typing import List
import uuid
from groq import Groq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings  # ✅ Updated import

app = FastAPI()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Configuration
EMBEDDING_MODEL = "sentence-transformers/paraphrase-albert-small-v2"  # ✅ Lighter model
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
MAX_DOCUMENTS = 20
MAX_PAGES = 1000

# Initialize ChromaDB with Hugging Face embeddings
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
vector_store = Chroma(embedding_function=embeddings, persist_directory="./chroma_db")

class DocumentRequest(BaseModel):
    text: str

class QueryRequest(BaseModel):
    question: str

def process_document(text: str, doc_id: str):
    try:
        # Dynamic chunking
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len
        )
        chunks = text_splitter.split_text(text)

        # Store in ChromaDB
        vector_store.add_texts(
            texts=chunks,
            ids=[f"{doc_id}_{i}" for i in range(len(chunks))],
            metadatas=[{"source": doc_id} for _ in chunks]
        )
    except Exception as e:
        print(f"Processing error: {str(e)}")

@app.post("/upload")
async def upload_document(file: UploadFile, background_tasks: BackgroundTasks):
    if len(vector_store.get()["ids"]) >= MAX_DOCUMENTS * 100:  # Estimate 100 chunks/doc
        raise HTTPException(400, "Document limit reached")

    content = await file.read()
    doc_id = str(uuid.uuid4())

    background_tasks.add_task(process_document, content.decode(), doc_id)
    return {"id": doc_id, "status": "processing"}

@app.post("/query")
async def query_documents(request: QueryRequest):
    try:
        # Retrieve relevant chunks
        results = vector_store.similarity_search(
            query=request.question,
            k=3  # ✅ Reduced for lower memory use
        )

        # Generate response with Groq
        context = "\n".join([doc.page_content for doc in results])
        response = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": f"Answer using context:\n{context}"},
                {"role": "user", "content": request.question}
            ],
            model="llama3-8b-8192",  # ✅ Smaller Groq model
            max_tokens=1024,
            temperature=0.3
        )

        return {"answer": response.choices[0].message.content}
    except Exception as e:
        raise HTTPException(500, f"Query error: {str(e)}")
