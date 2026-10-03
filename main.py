from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader
import faiss
import numpy as np
from google import genai
import io
import os
import time

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

RAG_STORE = {
    "index": None,
    "chunks": []
}

def get_embedding_with_retry(text_list: list[str], max_retries: int = 5) -> np.ndarray:
    """Generates embeddings with backoff retries to prevent 429 Rate Limit errors."""
    formatted_contents = [[text] for text in text_list]
    models_to_try = ["gemini-embedding-2-preview", "text-embedding-004"]
    
    for attempt in range(max_retries):
        for model_name in models_to_try:
            try:
                response = client.models.embed_content(
                    model=model_name,
                    contents=formatted_contents,
                )
                embeddings = [item.values for item in response.embeddings]
                return np.array(embeddings, dtype=np.float32)
            except Exception as e:
                # If rate limited (429 / RESOURCE_EXHAUSTED), wait and retry
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    wait_time = (2 ** attempt) + 1  # Waits 2s, 3s, 5s, 9s, 17s...
                    print(f"Rate limited (429). Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    break  # Retry loop
                continue
                
    raise HTTPException(status_code=429, detail="Exceeded API rate limits. Please wait 10 seconds and try again.")

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50):
    """Splits text into overlapping segments."""
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

        # Chunk text
        chunks = chunk_text(extracted_text)
        
        # Batch requests with a short delay to satisfy free-tier RPM limits
        batch_size = 20
        all_embeddings = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batch_emb = get_embedding_with_retry(batch)
            all_embeddings.append(batch_emb)
            time.sleep(1) # 1-second pause between batches
            
        embeddings = np.vstack(all_embeddings)
        
        # Build FAISS vector index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        
        RAG_STORE["index"] = index
        RAG_STORE["chunks"] = chunks
        
        return {
            "status": "success", 
            "message": f"Successfully processed {len(pdf_reader.pages)} pages ({len(chunks)} text chunks)!"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat_with_pdf(question: str = Form(...)):
    if RAG_STORE["index"] is None:
        raise HTTPException(status_code=400, detail="Please upload a PDF first.")
    
    query_embedding = get_embedding_with_retry([question])
    
    k = min(3, len(RAG_STORE["chunks"]))
    distances, indices = RAG_STORE["index"].search(query_embedding, k)
    
    retrieved_chunks = [RAG_STORE["chunks"][idx] for idx in indices[0] if idx < len(RAG_STORE["chunks"])]
    context = "\n\n---\n\n".join(retrieved_chunks)
    
    system_prompt = f"""
    You are an expert PDF assistant. 
    Answer the user's question accurately using ONLY the provided document text below. 
    Base your answer strictly on the provided document text. 
    If the answer cannot be found in the provided document text, state: "I couldn't find information about that in the provided document."

    --- PROVIDED DOCUMENT TEXT ---
    {context}
    ------------------------------
    """

    models_to_try = ["gemini-3.6-flash", "gemini-3.6-pro", "gemini-2.5-flash"]
    
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
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)