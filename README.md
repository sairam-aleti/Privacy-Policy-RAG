# 📖 RAG Pipeline Explanation

This project implements a **(RAG) pipeline** using **Ollama + FAISS + LangChain’s text splitter**.

---

## 🔹 1. Imports
- **faiss** → vector search engine for similarity search  
- **fitz (PyMuPDF)** → extract text from PDFs  
- **requests + BeautifulSoup** → fetch & parse website content  
- **numpy** → handle embeddings as vectors  
- **json** → parse API responses from Ollama  
- **Counter** → check most common embedding dimensions  
- **RecursiveCharacterTextSplitter** → split long text into smaller chunks  

---

## 🔹 2. Embeddings & Chat Functions

### `ollama_embed(texts, model="nomic-embed-text")`
- Sends text(s) to **Ollama embedding API** (`localhost:11434/api/embeddings`)  
- Returns a list of embedding vectors  

### `ollama_chat(contexts, question, model="llama2")`
- Joins retrieved chunks into a single **context**  
- Sends prompt (context + question) to **Ollama chat API**  
- Handles **streamed JSON responses** and builds the full answer  

---

## 🔹 3. Data Extraction

### `extract_text_from_pdf(pdf_path)`
- Opens PDF and extracts text from all pages  

### `extract_text_from_website(url)`
- Fetches webpage content  
- Parses with BeautifulSoup  
- Returns plain text (HTML stripped)  

---

## 🔹 4. Text Splitting

### `chunk_text(text, chunk_size=500, chunk_overlap=50)`
- Splits text into chunks of ~500 characters  
- Keeps **50-character overlap** for context continuity  
- Uses separators (`paragraphs → sentences → words`) for clean splits  

---

## 🔹 5. Embedding Cleanup

### `filter_embeddings_and_chunks(embeddings, chunks)`
- Ensures all embeddings have the **same dimension**  
- Keeps only embeddings of the most common length  
- Aligns valid embeddings with their corresponding chunks  

---

## 🔹 6. FAISS Index

### `create_faiss_index(embeddings)`
- Builds a **FAISS index** using L2 distance (Euclidean)  
- Stores embeddings for similarity search  

### `search_index(index, query_embedding, k=5)`
- Searches the index for **k nearest neighbors**  
- Returns indices of most relevant chunks  

---

## 🔹 7. Answer Generation

### `generate_answer(contexts, question)`
- Wrapper for `ollama_chat`  
- Takes retrieved contexts + user question → generates an answer  

---

## 🔹 8. App/Website Handling

### `read_apps_file(file_path)`
- Reads `apps.txt` in the format:  
  ```text
  app_name: https://example.com
  another_app: https://example.org


commands to run (make sure you have ollama installed):

ollama serve
curl http://localhost:11434/api/tags
ollama pull nomic-embed-text
python main.py