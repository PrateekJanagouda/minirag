import json
import numpy as np
import requests
import os

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


def ask(question:str,chunks:list[str],embeddings:list[np.ndarray], history:list= []) -> str:
    question_vec = get_embedding(question)

    scores = [cosine_similarity(question_vec,emb) for emb in embeddings]
    top_indices = np.argsort(scores)[-3:][::-1]
    top_chunks = [chunks[i] for i in top_indices]

    context = "\n\n".join(top_chunks)

    history_text = " "

    if history:
        history_text = "\n\n Previous Converstion:/n"

        for h in history[-3:]:
            history_text += f"User: {h['question']}\nAI: {h['answer']}\n"   

    prompt = f"""Answer the question based on the following context below.
    Context:
    {context}
    {history_text}
    Question: {question}
    Answer:"""


    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json = {"model":CHAT_MODEL,"prompt":prompt,"stream":False}
    )

    raw_anwer = response.text.strip().split("\n")[-1]
    return json.loads(raw_anwer)["response"]

def save_index(chunks:list[str], embeddings:list[np.ndarray]):
    np.save("embeddings.npy",embeddings)
    with open("chunks.json","w") as f:
        json.dump(chunks,f)
    print("Index saved to disk.")


def load_index():
    if os.path.exists("embeddings.npy") and os.path.exists("chunks.json"):
        embeddings = np.load("embeddings.npy")
        with open("chunks.json","r") as f:
            chunks = json.load(f)
        print("Index loaded from disk.")
        return chunks, embeddings
    return None, None
        
if __name__ == "__main__":

    chunks, embeddings = load_index()

    if chunks is None :
        with open("sample.txt", "r") as f:
            text = f.read()

        print("Chunking text...")
        chunks = chunk_text(text)
        print(f"Created {len(chunks)} chunks")

        print("Embedding chunks...")
        embeddings = [get_embedding(chunk) for chunk in chunks]
        
        save_index(chunks, embeddings)

    print("Done! Ask me anything.\n")


    history = []

    while True:
        question = input("You: ").strip()
        if not question:
            break
        answer = ask(question, chunks, embeddings,history)
        print(f"AI: {answer}\n")

        history.append({"question":question,"answer":answer})