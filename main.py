import faiss
import fitz  # PyMuPDF
import requests
from bs4 import BeautifulSoup
import numpy as np
import json
from collections import Counter
from langchain.text_splitter import RecursiveCharacterTextSplitter


def ollama_embed(texts, model="nomic-embed-text"):
    url = "http://localhost:11434/api/embeddings"
    embeddings = []
    for text in texts:
        response = requests.post(url, json={"model": model, "prompt": text})
        response.raise_for_status()
        embeddings.append(response.json()["embedding"])
    return embeddings

def ollama_chat(contexts, question, model="llama2"):
    url = "http://localhost:11434/api/chat"
    context = "\n\n".join(contexts)
    prompt = f"Given the following context, answer the question:\n\nContext:\n{context}\n\nQuestion:\n{question}"
    response = requests.post(
        url,
        json={"model": model, "messages": [{"role": "user", "content": prompt}]},
        stream=True
    )
    response.raise_for_status()
    full_response = ""
    for line in response.iter_lines():
        if line:
            data = line.decode("utf-8")
            try:
                chunk = json.loads(data)
                if "message" in chunk and "content" in chunk["message"]:
                    full_response += chunk["message"]["content"]
                elif "content" in chunk:
                    full_response += chunk["content"]
            except Exception:
                continue
    return full_response

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def extract_text_from_website(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        return soup.get_text(separator=" ", strip=True)
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return ""

def chunk_text(text, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return splitter.split_text(text)

def filter_embeddings_and_chunks(embeddings, chunks):
    lengths = [len(e) for e in embeddings]
    most_common_length = Counter(lengths).most_common(1)[0][0]
    filtered_embeddings = []
    filtered_chunks = []
    for emb, chunk in zip(embeddings, chunks):
        if len(emb) == most_common_length:
            filtered_embeddings.append(emb)
            filtered_chunks.append(chunk)
    return filtered_embeddings, filtered_chunks

def embed_text(texts):
    return ollama_embed(texts)

def create_faiss_index(embeddings):
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    return index

def search_index(index, query_embedding, k=5):
    D, I = index.search(np.array([query_embedding]).astype('float32'), k)
    return I[0]

def generate_answer(contexts, question):
    return ollama_chat(contexts, question)

def read_apps_file(file_path):
    apps = {}
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and ':' in line:
                app_name, url = line.split(':', 1)
                apps[app_name.strip()] = url.strip()
    return apps

def process_all_apps(apps_file):
    apps = read_apps_file(apps_file)
    
    all_chunks = []
    chunk_sources = []  
    
    for app_name, url in apps.items():
        print(f"Processing {app_name}...")
        text = extract_text_from_website(url)
        if text:
            chunks = chunk_text(text)
            all_chunks.extend(chunks)
            chunk_sources.extend([app_name] * len(chunks))
    
    return all_chunks, chunk_sources

if __name__ == "__main__":
    chunks, chunk_sources = process_all_apps("apps.txt")
    
    print("Creating embeddings...")
    embeddings = embed_text(chunks)
    embeddings, chunks = filter_embeddings_and_chunks(embeddings, chunks)
    index = create_faiss_index(embeddings)
    
    print("\nRAG Pipeline is ready! Type 'quit' to exit.")
    while True:
        user_question = input("\nEnter your question: ")
        if user_question.lower() == 'quit':
            break
            
        query_embedding = embed_text([user_question])[0]
        top_k = search_index(index, query_embedding, k=3)
        relevant_chunks = [chunks[i] for i in top_k]
        
        answer = generate_answer(relevant_chunks, user_question)
        print("\nAnswer:", answer)
        
        print("\nSources:")
        for i in top_k:
            print(f"- {chunk_sources[i]}")
