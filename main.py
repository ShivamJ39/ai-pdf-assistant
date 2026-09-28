from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from google import genai
import io
import os

app = FastAPI()

# Enable CORS for deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Fetch API key from Environment Variable (security best practice)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", )
client = genai.Client(api_key=GEMINI_API_KEY)

# Load embedding model for vector search
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# In-memory vector store
RAG_STORE = {
    "index": None,
    "chunks": []
}

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50):
    """Splits text into overlapping chunks for RAG."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks

@app.get("/")
def read_root():
    return {"status": "PDF AI Assistant API is running"}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    try:
        contents = await file.read()
        pdf_reader = PdfReader(io.BytesIO(contents))
        extracted_text = ""
        
        for page in pdf_reader.pages:
            extracted_text += page.extract_text() or ""
            
        if not extracted_text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from PDF.")

        # Build Vector Store (RAG)
        chunks = chunk_text(extracted_text)
        embeddings = embedder.encode(chunks, convert_to_numpy=True)
        
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings, dtype=np.float32))
        
        RAG_STORE["index"] = index
        RAG_STORE["chunks"] = chunks
        
        return {
            "status": "success", 
            "message": f"Successfully Processed {len(pdf_reader.pages)} pages "
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat_with_pdf(question: str = Form(...)):
    if RAG_STORE["index"] is None:
        raise HTTPException(status_code=400, detail="Please upload a PDF first.")
    
    # Retrieve top 3 relevant text chunks
    query_embedding = embedder.encode([question], convert_to_numpy=True)
    k = min(3, len(RAG_STORE["chunks"]))
    distances, indices = RAG_STORE["index"].search(np.array(query_embedding, dtype=np.float32), k)
    
    retrieved_chunks = [RAG_STORE["chunks"][idx] for idx in indices[0] if idx < len(RAG_STORE["chunks"])]
    context = "\n\n---\n\n".join(retrieved_chunks)
    
    system_prompt = f"""
    You are an expert RAG PDF assistant. Answer the user's question accurately using ONLY 
    the provided retrieved document text below. If the answer is not in the context, 
    say "I couldn't find this information in the document."

    --- RETRIEVED CONTEXT ---
    {context}
    ------------------------
    """

    models_to_try = [ "gemini-3.6-flash" ]
    
    for model in models_to_try:
        try:
            response = client.models.generate_content(
                model=model,
                contents=[system_prompt, f"User Question: {question}"]
            )
            return {"answer": response.text}
        except Exception as e:
            if "503" in str(e) or "UNAVAILABLE" in str(e):
                continue
            raise HTTPException(status_code=500, detail=f"AI API Error: {str(e)}")

    raise HTTPException(status_code=503, detail="AI service busy, please try again shortly.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)