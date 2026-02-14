import os
import re
import json
import time
import hashlib
from typing import Dict, List, Tuple, Any

import faiss
import fitz  # PyMuPDF
import requests
from bs4 import BeautifulSoup
import numpy as np
from collections import Counter
from langchain.text_splitter import RecursiveCharacterTextSplitter


# ----------------------------
# Ollama API helpers
# ----------------------------

OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"

EMBED_MODEL = "nomic-embed-text"

# CHANGED: smaller chat model to avoid RAM error seen with llama2
CHAT_MODEL = "llama3.2:3b"


def ollama_embed(texts: List[str], model: str = EMBED_MODEL, timeout: int = 120) -> List[List[float]]:
    """
    Calls Ollama embeddings endpoint. One request per text (simple + reliable).
    """
    embeddings = []
    for idx, text in enumerate(texts, start=1):
        resp = requests.post(
            OLLAMA_EMBED_URL,
            json={"model": model, "prompt": text},
            timeout=timeout,
        )
        resp.raise_for_status()
        embeddings.append(resp.json()["embedding"])
    return embeddings


def ollama_chat(contexts_labeled: List[str], question: str, model: str = CHAT_MODEL, timeout: int = 300) -> str:
    """
    Calls Ollama chat endpoint with a strict "cite-your-evidence" instruction.
    contexts_labeled: list like ["[app#C12] chunk...", "[app#C57] chunk..."]
    """
    context = "\n\n".join(contexts_labeled)

    system_msg = (
        "You are a strict privacy-policy auditor.\n"
        "Rules:\n"
        "1) Use ONLY the provided Context. Do NOT use outside knowledge.\n"
        "2) Every factual claim MUST include one or more citations to chunk IDs "
        "exactly as they appear in the Context (e.g., [aarogya_setu#C12]).\n"
        "3) If the Context does not contain enough evidence, say 'Insufficient evidence in provided context.'\n"
        "4) Do NOT claim 'X not found in policy' unless you can justify it from the Context.\n"
        "5) Prefer quoting exact phrases from the Context when possible.\n"
    )

    user_msg = (
        "Given the following Context, answer the Question.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n"
    )

    resp = requests.post(
        OLLAMA_CHAT_URL,
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
        },
        stream=True,
        timeout=timeout,
    )
    resp.raise_for_status()

    full_response = ""
    for line in resp.iter_lines():
        if not line:
            continue
        try:
            chunk = json.loads(line.decode("utf-8"))
        except Exception:
            continue

        # Ollama streaming responses usually include {"message":{"content":"..."}, ...}
        if "message" in chunk and isinstance(chunk["message"], dict) and "content" in chunk["message"]:
            full_response += chunk["message"]["content"]
        elif "content" in chunk:
            full_response += chunk["content"]

    return full_response.strip()


# ----------------------------
# App name normalization (for matching findings with apps)
# ----------------------------

def app_key(name: str) -> str:
    """
    Canonical app identifier for matching when names vary.
    Examples:
      "Aarogya Setu" -> "aarogyasetu"
      "AarogyaSetu"  -> "aarogyasetu"
    """
    return re.sub(r"[^a-z0-9]+", "", name.strip().lower())


def slugify(name: str) -> str:
    """
    Safe folder name.
    Examples:
      "Aarogya Setu" -> "aarogya_setu"
      "CoinDCX Web3" -> "coindcx_web3"
    """
    s = name.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unknown_app"


# ----------------------------
# Text extraction
# ----------------------------

def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    out = []
    for page in doc:
        out.append(page.get_text())
    return "\n".join(out)


def extract_text_from_website(url: str, timeout: int = 30) -> Tuple[str, str]:
    """
    Returns (text, source_type) where source_type is 'html' or 'pdf' or 'unknown'.
    Detects PDF via URL suffix OR response content-type.
    """
    try:
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()

        content_type = (resp.headers.get("Content-Type") or "").lower()
        is_pdf = url.lower().endswith(".pdf") or ("application/pdf" in content_type)

        if is_pdf:
            text = extract_text_from_pdf_bytes(resp.content)
            return text, "pdf"

        soup = BeautifulSoup(resp.text, "html.parser")
        text = soup.get_text(separator=" ", strip=True)
        return text, "html"

    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return "", "unknown"


# ----------------------------
# Chunking + embeddings + FAISS
# ----------------------------

def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)


def filter_embeddings_and_chunks(embeddings: List[List[float]], chunks: List[str]) -> Tuple[List[List[float]], List[str]]:
    """
    Ensures embedding vectors all have the same dimension (keeps the most common).
    """
    lengths = [len(e) for e in embeddings]
    most_common_length = Counter(lengths).most_common(1)[0][0]
    filtered_embeddings, filtered_chunks = [], []
    for emb, chunk in zip(embeddings, chunks):
        if len(emb) == most_common_length:
            filtered_embeddings.append(emb)
            filtered_chunks.append(chunk)
    return filtered_embeddings, filtered_chunks


def create_faiss_index(embeddings: List[List[float]]) -> faiss.Index:
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))
    return index


def search_index(index: faiss.Index, query_embedding: List[float], k: int = 5) -> List[int]:
    D, I = index.search(np.array([query_embedding]).astype("float32"), k)
    return I[0].tolist()


# ----------------------------
# Persistent per-app store
# ----------------------------

STORE_ROOT = "stores"


def app_store_dir(app_slug: str) -> str:
    return os.path.join(STORE_ROOT, app_slug)


def app_index_path(app_slug: str) -> str:
    return os.path.join(app_store_dir(app_slug), "index.faiss")


def app_chunks_path(app_slug: str) -> str:
    return os.path.join(app_store_dir(app_slug), "chunks.json")


def app_meta_path(app_slug: str) -> str:
    return os.path.join(app_store_dir(app_slug), "meta.json")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def compute_text_fingerprint(text: str) -> str:
    """
    Fingerprint the extracted text so we can detect policy changes and rebuild.
    """
    h = hashlib.sha256()
    h.update(text.encode("utf-8", errors="ignore"))
    return h.hexdigest()


def save_app_store(app_slug: str, chunks: List[str], index: faiss.Index, meta: Dict[str, Any]) -> None:
    ensure_dir(app_store_dir(app_slug))
    faiss.write_index(index, app_index_path(app_slug))
    with open(app_chunks_path(app_slug), "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    with open(app_meta_path(app_slug), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def load_app_store(app_slug: str) -> Tuple[List[str], faiss.Index, Dict[str, Any]]:
    idx_path = app_index_path(app_slug)
    ch_path = app_chunks_path(app_slug)
    m_path = app_meta_path(app_slug)

    if not (os.path.exists(idx_path) and os.path.exists(ch_path) and os.path.exists(m_path)):
        raise FileNotFoundError("App store not found / incomplete.")

    index = faiss.read_index(idx_path)
    with open(ch_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    with open(m_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    return chunks, index, meta


def build_or_load_app_store(app_name: str, url: str, rebuild: bool = False) -> Tuple[str, List[str], faiss.Index, Dict[str, Any]]:
    """
    Returns (app_slug, chunks, index, meta)
    """
    app_slug = slugify(app_name)
    ensure_dir(STORE_ROOT)
    ensure_dir(app_store_dir(app_slug))

    text, source_type = extract_text_from_website(url)
    if not text:
        raise RuntimeError(f"No text extracted for {app_name} ({url}).")

    fingerprint = compute_text_fingerprint(text)

    # Try to load existing store unless rebuild requested
    if not rebuild:
        try:
            chunks_existing, index_existing, meta_existing = load_app_store(app_slug)
            if meta_existing.get("fingerprint") == fingerprint:
                return app_slug, chunks_existing, index_existing, meta_existing
            else:
                print(f"Policy text changed for {app_name} -> rebuilding index.")
        except Exception:
            pass  # build below

    # Build new store
    chunks = chunk_text(text)
    print(f"  Chunks for {app_name}: {len(chunks)} (source={source_type})")
    print(f"  Embedding chunks for {app_name}...")

    embeddings = ollama_embed(chunks, model=EMBED_MODEL)
    embeddings, chunks = filter_embeddings_and_chunks(embeddings, chunks)
    index = create_faiss_index(embeddings)

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

    save_app_store(app_slug, chunks, index, meta)
    return app_slug, chunks, index, meta


# ----------------------------
# apps.txt parsing
# ----------------------------

def read_apps_file(file_path: str) -> Dict[str, Dict[str, str]]:
    """
    Returns mapping:
      app_key -> {"app_name": <display>, "url": <url>, "slug": <slug>}
    """
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


# ----------------------------
# Main
# ----------------------------

if __name__ == "__main__":
    apps = read_apps_file("apps.txt")

    if not apps:
        print("No apps found in apps.txt")
        raise SystemExit(1)

    print(f"Found {len(apps)} apps in apps.txt")
    print("Building/loading per-app FAISS stores...")

    # app_key -> store objects
    stores: Dict[str, Dict[str, Any]] = {}

    for k, info in apps.items():
        app_name = info["app_name"]
        url = info["url"]
        print(f"\nProcessing {app_name}...")
        try:
            app_slug, chunks, index, meta = build_or_load_app_store(app_name, url, rebuild=False)
            stores[k] = {
                "app_name": app_name,
                "app_slug": app_slug,
                "url": url,
                "chunks": chunks,
                "index": index,
                "meta": meta,
            }
            print(f"  Store ready: stores/{app_slug}/ (chunks={len(chunks)})")
        except Exception as e:
            print(f"  Skipping {app_name} due to error: {e}")

    if not stores:
        print("No app stores could be built/loaded. Exiting.")
        raise SystemExit(1)

    print("\nRAG is ready. Type 'list' to see apps, 'quit' to exit.")
    while True:
        raw_app = input("\nEnter app name for this query (or 'list'/'quit'): ").strip()
        if raw_app.lower() == "quit":
            break
        if raw_app.lower() == "list":
            print("\nAvailable apps:")
            for info in stores.values():
                print(f"- {info['app_name']}")
            continue

        k = app_key(raw_app)
        if k not in stores:
            print("App not found. Try 'list' and copy the exact name, or enter a close variant.")
            continue

        user_question = input("Enter your question/finding: ").strip()
        if user_question.lower() == "quit":
            break

        store = stores[k]
        app_slug = store["app_slug"]
        chunks = store["chunks"]
        index = store["index"]

        # Retrieve
        query_embedding = ollama_embed([user_question], model=EMBED_MODEL)[0]
        top_k = search_index(index, query_embedding, k=5)

        # Build labeled contexts
        contexts_labeled = []
        evidence_items = []
        for idx in top_k:
            cid = f"{app_slug}#C{idx}"
            chunk_text_str = chunks[idx]
            contexts_labeled.append(f"[{cid}] {chunk_text_str}")
            evidence_items.append((cid, chunk_text_str))

        # Generate
        answer = ollama_chat(contexts_labeled, user_question, model=CHAT_MODEL)

        print("\nAnswer:\n", answer)

        print("\nEvidence (retrieved chunks):")
        for cid, ctext in evidence_items:
            print(f"\n[{cid}]\n{ctext}")

        print("\nSource:")
        print(f"- {store['app_name']} ({store['url']})")