# MiniRAG 

A minimal RAG system built from scratch using pure Python.
No LangChain. No ChromaDB. Just numpy and Ollama.

## Endpoints
- POST /ingest — chunk and embed any text
- POST /query — ask questions against ingested data
- GET /health — system status

## Tech Stack
Python | FastAPI | Ollama | nomic-embed-text | numpy

## Run
pip install -r requirements.txt
uvicorn api:app --reload