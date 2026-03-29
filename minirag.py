import json
import numpy as np
import requests

OLLAMA_URL = "http://10.169.51.167:11434"
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3.2:latest"




def get_embedding(text:str) -> np.ndarray:
    response = requests.post(
                f"{OLLAMA_URL}/api/embeddings",
            json = {"model":EMBED_MODEL,"prompt":text}
    )

    return np.array(response.json()['embedding'])



def cosine_similarity(a:np.ndarray,b:np.ndarray) -> float :
    return np.dot(a,b) / (np.linalg.norm(a) * np.linalg.norm(b))



def chunk_text(text:str ,chunk_size:int = 1000 ,overlap:int = 50) -> list[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


def ask(question:str,chunks:list[str],embeddings:list[np.ndarray]) -> str:
    question_vec = get_embedding(question)

    scores = [cosine_similarity(question_vec,emb) for emb in embeddings]
    top_indices = np.argsort(scores)[-3:][::-1]
    top_chunks = [chunks[i] for i in top_indices]

    context = "\n\n".join(top_chunks)

    prompt = f"""Answer the question based on the following context below.
    Context:
    {context}
    Question: {question}
    Answer:"""


    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json = {"model":CHAT_MODEL,"prompt":prompt,"stream":False}
    )

    raw_anwer = response.text.strip().split("\n")[-1]
    return json.loads(raw_anwer)["response"]

if __name__ == "__main__":
    # Load any text file
    with open("sample.txt", "r") as f:
        text = f.read()

    print("Chunking text...")
    chunks = chunk_text(text)
    print(f"Created {len(chunks)} chunks")

    print("Embedding chunks...")
    embeddings = [get_embedding(chunk) for chunk in chunks]
    print("Done! Ask me anything.\n")

    while True:
        question = input("You: ").strip()
        if not question:
            break
        answer = ask(question, chunks, embeddings)
        print(f"AI: {answer}\n")