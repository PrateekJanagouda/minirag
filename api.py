from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import numpy as np
from minirag import get_embedding, chunk_text, ask



app = FastAPI(title="MiniRAG app")


#store the chunks and their embeddings in memory for simplicity

store = {  
    "chunks":[],
    "embeddings":[]
}


class Ingestrequest(BaseModel):
    text:str

class QueryRequest(BaseModel):
    question:str


@app.post("/ingest")
def ingest(request:Ingestrequest):
    chunks = chunk_text(request.text)
    embeddings = [get_embedding(chunk) for chunk in chunks]

    store["chunks"].extend(chunks)
    store["embeddings"].extend(embeddings)

    return {"message":f"Ingested {len(chunks)} chunks"}

@app.post("/query")
def query(request:QueryRequest):
    if not store["chunks"]:
        return JSONResponse(content={"error":"No data ingested yet"},status_code=400)
    
    answer = ask(request.question,store["chunks"],store["embeddings"])
    return {"answer":answer}


@app.get("/health")
def health():
    return {"status": "ok", "chunks_loaded": len(store["chunks"])}



