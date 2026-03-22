import os
import re
import json
import faiss  # type: ignore
import numpy as np  # type: ignore
import hashlib
import time
from typing import List, Tuple, Dict, Any
import pickle
from rank_bm25 import BM25Okapi  # type: ignore

from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore
from .llm import ollama_embed, EMBED_MODEL, CHAT_MODEL  # type: ignore
from .scraper import extract_text_from_website  # type: ignore

def app_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.strip().lower())

def slugify(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unknown_app"

def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)

def filter_embeddings_and_chunks(embeddings: List[List[float]], chunks: List[str]) -> Tuple[List[List[float]], List[str]]:
    if not embeddings:
        return [], []
    length_counts: Dict[int, int] = {}
    for e in embeddings:
        length_counts[len(e)] = length_counts.get(len(e), 0) + 1
    most_common_length = max(length_counts.keys(), key=lambda k: length_counts[k])
    
    filtered_embeddings: List[List[float]] = []
    filtered_chunks: List[str] = []
    for i in range(len(embeddings)):
        if len(embeddings[i]) == most_common_length:
            filtered_embeddings.append(embeddings[i])
            filtered_chunks.append(chunks[i])
    return filtered_embeddings, filtered_chunks

def create_faiss_index(embeddings: List[List[float]]) -> faiss.Index:
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))
    return index

def search_index(index: faiss.Index, query_embedding: List[float], k: int = 20) -> List[int]:
    D, I = index.search(np.array([query_embedding]).astype("float32"), k)
    return I[0].tolist()

STORE_ROOT = "stores"

def app_store_dir(app_slug: str) -> str:
    return os.path.join(STORE_ROOT, app_slug)

def app_index_path(app_slug: str) -> str:
    return os.path.join(app_store_dir(app_slug), "index.faiss")

def app_bm25_path(app_slug: str) -> str:
    return os.path.join(app_store_dir(app_slug), "bm25.pkl")

def app_chunks_path(app_slug: str) -> str:
    return os.path.join(app_store_dir(app_slug), "chunks.json")

def app_meta_path(app_slug: str) -> str:
    return os.path.join(app_store_dir(app_slug), "meta.json")

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def compute_text_fingerprint(text: str) -> str:
    h = hashlib.sha256()
    h.update(text.encode("utf-8", errors="ignore"))
    return h.hexdigest()

def save_app_store(app_slug: str, chunks: List[str], index: faiss.Index, bm25: BM25Okapi, meta: Dict[str, Any]) -> None:
    ensure_dir(app_store_dir(app_slug))
    faiss.write_index(index, app_index_path(app_slug))
    with open(app_bm25_path(app_slug), "wb") as f:
        pickle.dump(bm25, f)
    with open(app_chunks_path(app_slug), "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    with open(app_meta_path(app_slug), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

def load_app_store(app_slug: str) -> Tuple[List[str], faiss.Index, BM25Okapi, Dict[str, Any]]:
    idx_path = app_index_path(app_slug)
    bm25_path = app_bm25_path(app_slug)
    ch_path = app_chunks_path(app_slug)
    m_path = app_meta_path(app_slug)

    if not (os.path.exists(idx_path) and os.path.exists(bm25_path) and os.path.exists(ch_path) and os.path.exists(m_path)):
        raise FileNotFoundError("App store not found / incomplete.")

    index = faiss.read_index(idx_path)
    with open(bm25_path, "rb") as f:
        bm25 = pickle.load(f)
    with open(ch_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    with open(m_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    return chunks, index, bm25, meta

def build_or_load_app_store(app_name: str, url: str, rebuild: bool = False) -> Tuple[str, List[str], faiss.Index, BM25Okapi, Dict[str, Any]]:
    app_slug = slugify(app_name)
    ensure_dir(STORE_ROOT)
    ensure_dir(app_store_dir(app_slug))

    text, source_type = extract_text_from_website(url)
    if not text:
        raise RuntimeError(f"No text extracted for {app_name} ({url}).")

    fingerprint = compute_text_fingerprint(text)

    if not rebuild:
        try:
            chunks_existing, index_existing, bm25_existing, meta_existing = load_app_store(app_slug)
            if meta_existing.get("fingerprint") == fingerprint:
                return app_slug, chunks_existing, index_existing, bm25_existing, meta_existing
            else:
                print(f"Policy text changed for {app_name} -> rebuilding index.")
        except Exception:
            pass

    chunks = chunk_text(text)
    print(f"  Chunks for {app_name}: {len(chunks)} (source={source_type})")
    print(f"  Embedding chunks for {app_name}...")

    embeddings = ollama_embed(chunks, model=EMBED_MODEL)
    embeddings, chunks = filter_embeddings_and_chunks(embeddings, chunks)
    index = create_faiss_index(embeddings)
    
    # Build BM25
    tokenized_corpus = [c.lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    meta = {
        "app_name": app_name,
        "app_slug": app_slug,
        "url": url,
        "source_type": source_type,
        "fingerprint": fingerprint,
        "chunk_size": 500,
        "chunk_overlap": 50,
        "embed_model": EMBED_MODEL,
        "chat_model": CHAT_MODEL,
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    save_app_store(app_slug, chunks, index, bm25, meta)
    return app_slug, chunks, index, bm25, meta

def read_apps_file(file_path: str) -> Dict[str, Dict[str, str]]:
    apps: Dict[str, Dict[str, str]] = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            app_name, url = line.split(":", 1)
            app_name = app_name.strip()
            url = url.strip()
            k = app_key(app_name)
            apps[k] = {"app_name": app_name, "url": url, "slug": slugify(app_name)}
    return apps
