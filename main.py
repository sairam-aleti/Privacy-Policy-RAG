import os
import re
import json
import time
import hashlib
from typing import Dict, List, Tuple, Any, Set

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
CHAT_MODEL = "llama3.2:3b"  # smaller model (llama2 failed due to RAM)


def ollama_embed(texts: List[str], model: str = EMBED_MODEL, timeout: int = 120) -> List[List[float]]:
    embeddings = []
    for text in texts:
        resp = requests.post(
            OLLAMA_EMBED_URL,
            json={"model": model, "prompt": text},
            timeout=timeout,
        )
        resp.raise_for_status()
        embeddings.append(resp.json()["embedding"])
    return embeddings


def ollama_chat(contexts_labeled: List[str], question: str, model: str = CHAT_MODEL, timeout: int = 300) -> str:
    context = "\n\n".join(contexts_labeled)

    system_msg = (
        "You are a strict privacy-policy auditor.\n"
        "Rules:\n"
        "1) Use ONLY the provided Context. Do NOT use outside knowledge.\n"
        "2) Every bullet/factual claim MUST end with one or more citations to chunk IDs "
        "exactly as they appear in the Context (e.g., [paytm#C11]).\n"
        "3) If the Context does not contain enough evidence for a claim, say "
        "'Insufficient evidence in provided context.'\n"
        "4) Do NOT claim 'X not found in policy'. Only speak about what is present in Context.\n"
        "5) Do NOT invent citation IDs.\n"
    )

    user_msg = (
        "Answer the Question using only the Context.\n"
        "Return 2-6 bullet points. Each bullet must end with citations.\n\n"
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
        if "message" in chunk and isinstance(chunk["message"], dict) and "content" in chunk["message"]:
            full_response += chunk["message"]["content"]
        elif "content" in chunk:
            full_response += chunk["content"]

    return full_response.strip()


# ----------------------------
# Term detection (deterministic, for now)
# Later: replace with structured findings -> data_type
# ----------------------------

PRIVACY_TERMS = {
    "location": ["location", "gps", "geolocation", "precise location", "lat", "lon", "latitude", "longitude"],
    "imei": ["imei"],
    "aadhaar": ["aadhaar", "aadhar"],
    "advertising_id": ["advertising id", "ad id", "aaid", "google advertising id"],
    "android_id": ["android id"],
    "device_id": ["device id", "device identifier", "unique identifier", "hardware id"],
    "mac_address": ["mac address", "mac"],
    "ip_address": ["ip address", "ip"],
    "contacts": ["contacts", "address book"],
    "sms": ["sms", "text messages"],
    "camera": ["camera"],
    "microphone": ["microphone", "mic"],
    "email": ["email", "email id", "e-mail"],
    "phone_number": ["phone number", "mobile number", "phone"],
    "name": ["name"],
    "age": ["age", "date of birth", "dob"],
    "sex": ["sex", "gender"],
    "profession": ["profession", "occupation"],
    "payment": ["payment", "card", "credit card", "debit card", "upi", "bank", "account number"],
    "health": ["health", "medical", "diagnosis", "treatment"],
    "cookies": ["cookie", "cookies"],
}

MENTION_TRIGGERS = [
    "mention", "mentions", "mentioned",
    "contain", "contains", "containing",
    "explicit", "explicitly",
    "state", "states", "stated",
    "refer", "refers", "reference", "references",
    "list", "lists", "listed",
]


def normalize_question(q: str) -> str:
    return (
        q.replace("“", '"')
         .replace("”", '"')
         .replace("’", "'")
         .replace("‘", "'")
         .strip()
    )


def is_mention_query(question: str) -> bool:
    q = question.lower()
    return any(t in q for t in MENTION_TRIGGERS)


def detect_terms(question: str) -> List[str]:
    q = question.lower()
    found = []
    for canon, syns in PRIVACY_TERMS.items():
        for s in syns:
            if s in q:
                found.append(canon)
                break
    return found


def synonyms_for(terms: List[str]) -> List[str]:
    syns: List[str] = []
    for t in terms:
        syns.extend(PRIVACY_TERMS.get(t, []))
    syns = sorted(set(syns), key=len, reverse=True)
    return syns


def build_matchers(syns: List[str]):
    matchers = []
    for s in syns:
        s = s.strip().lower()
        if not s:
            continue
        if len(s) <= 3 and " " not in s:
            matchers.append(("regex", re.compile(rf"\b{re.escape(s)}\b", re.IGNORECASE)))
        else:
            matchers.append(("substr", s))
    return matchers


def chunk_contains_any(text: str, matchers) -> bool:
    low = text.lower()
    for kind, pat in matchers:
        if kind == "substr":
            if pat in low:
                return True
        else:
            if pat.search(text):
                return True
    return False


def extract_matching_sentences(chunk: str, matchers) -> List[str]:
    parts = re.split(r"(?<=[\.\?\!])\s+|\n+", chunk)
    hits = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if chunk_contains_any(p, matchers):
            hits.append(p)
    if not hits and chunk_contains_any(chunk, matchers):
        hits = [chunk.strip()]
    return hits


# ----------------------------
# Citation validation (NEW)
# ----------------------------

_CIT_RE = re.compile(r"\[([^\[\]]+)\]")  # extracts inside brackets


def extract_citation_ids(answer_text: str) -> List[str]:
    return _CIT_RE.findall(answer_text or "")


def is_insufficient_evidence_answer(answer_text: str) -> bool:
    if not answer_text:
        return True
    return "insufficient evidence" in answer_text.lower()


def validate_llm_answer(answer_text: str, allowed_ids: Set[str]) -> Tuple[bool, Dict[str, Any]]:
    """
    Strict validator:
    - Any bracketed citation must be in allowed_ids.
    - If answer has bullet lines, each bullet must contain >=1 allowed citation.
    - If answer is an 'Insufficient evidence...' refusal, accept it (no citations required).
    """
    report: Dict[str, Any] = {
        "allowed_ids": sorted(list(allowed_ids)),
        "all_bracket_ids": [],
        "invalid_citations": [],
        "bullets_checked": 0,
        "bullets_missing_valid_citation": 0,
    }

    if is_insufficient_evidence_answer(answer_text):
        report["reason"] = "model_refused_insufficient_evidence"
        return True, report

    all_ids = extract_citation_ids(answer_text)
    report["all_bracket_ids"] = all_ids

    invalid = [cid for cid in all_ids if cid not in allowed_ids]
    report["invalid_citations"] = invalid
    if invalid:
        report["reason"] = "invalid_citation_ids_present"
        return False, report

    lines = [ln.strip() for ln in (answer_text or "").splitlines() if ln.strip()]
    bullet_lines = [ln for ln in lines if ln.startswith(("-", "*", "•"))]

    if bullet_lines:
        report["bullets_checked"] = len(bullet_lines)
        for ln in bullet_lines:
            ids_in_line = extract_citation_ids(ln)
            has_valid = any(cid in allowed_ids for cid in ids_in_line)
            if not has_valid:
                report["bullets_missing_valid_citation"] += 1
        if report["bullets_missing_valid_citation"] > 0:
            report["reason"] = "some_bullets_missing_valid_citation"
            return False, report

    # No bullets: require at least one valid citation somewhere (otherwise it's ungrounded narrative)
    if not bullet_lines and len(all_ids) == 0:
        report["reason"] = "no_citations_present"
        return False, report

    report["reason"] = "ok"
    return True, report


# ----------------------------
# App name normalization
# ----------------------------

def app_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.strip().lower())


def slugify(name: str) -> str:
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
    try:
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()

        content_type = (resp.headers.get("Content-Type") or "").lower()
        is_pdf = url.lower().endswith(".pdf") or ("application/pdf" in content_type)

        if is_pdf:
            return extract_text_from_pdf_bytes(resp.content), "pdf"

        soup = BeautifulSoup(resp.text, "html.parser")
        return soup.get_text(separator=" ", strip=True), "html"

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


def search_index(index: faiss.Index, query_embedding: List[float], k: int = 20) -> List[int]:
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
    app_slug = slugify(app_name)
    ensure_dir(STORE_ROOT)
    ensure_dir(app_store_dir(app_slug))

    text, source_type = extract_text_from_website(url)
    if not text:
        raise RuntimeError(f"No text extracted for {app_name} ({url}).")

    fingerprint = compute_text_fingerprint(text)

    if not rebuild:
        try:
            chunks_existing, index_existing, meta_existing = load_app_store(app_slug)
            if meta_existing.get("fingerprint") == fingerprint:
                return app_slug, chunks_existing, index_existing, meta_existing
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
# Evidence selection logic
# ----------------------------

K_SEMANTIC = 20
K_RETURN = 3
MAX_EXPLICIT = 2


def select_evidence(chunks: List[str], index: faiss.Index, app_slug: str, question: str) -> Tuple[List[Tuple[str, str]], Dict[str, Any]]:
    terms = detect_terms(question)
    syns = synonyms_for(terms)
    matchers = build_matchers(syns)

    query_embedding = ollama_embed([question], model=EMBED_MODEL)[0]
    cand_indices = search_index(index, query_embedding, k=K_SEMANTIC)

    explicit_in_candidates = []
    if syns:
        for idx in cand_indices:
            if chunk_contains_any(chunks[idx], matchers):
                explicit_in_candidates.append(idx)

    explicit_global = []
    if syns:
        for i, ch in enumerate(chunks):
            if chunk_contains_any(ch, matchers):
                explicit_global.append(i)

    chosen = []
    explicit_chosen = []

    for idx in explicit_in_candidates:
        if idx not in chosen:
            chosen.append(idx)
            explicit_chosen.append(idx)
        if len(explicit_chosen) >= MAX_EXPLICIT:
            break

    if len(explicit_chosen) < MAX_EXPLICIT:
        for idx in explicit_global:
            if idx not in chosen:
                chosen.append(idx)
                explicit_chosen.append(idx)
            if len(explicit_chosen) >= MAX_EXPLICIT:
                break

    for idx in cand_indices:
        if len(chosen) >= K_RETURN:
            break
        if idx not in chosen:
            chosen.append(idx)

    evidence_items = []
    for idx in chosen[:K_RETURN]:
        cid = f"{app_slug}#C{idx}"
        evidence_items.append((cid, chunks[idx]))

    debug = {
        "detected_terms": terms,
        "synonyms_used": syns,
        "explicit_found_anywhere": bool(explicit_global),
        "explicit_global_indices": explicit_global[:50],
        "explicit_global_count": len(explicit_global),
        "explicit_selected_count": len(explicit_chosen),
        "semantic_candidate_count": len(cand_indices),
    }
    return evidence_items, debug


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

        question = normalize_question(input("Enter your question/finding: ").strip())
        if question.lower() == "quit":
            break

        store = stores[k]
        app_slug = store["app_slug"]
        chunks = store["chunks"]
        index = store["index"]

        evidence_items, debug = select_evidence(chunks, index, app_slug, question)

        print("\nClosest evidence (top 3 chunks):")
        for cid, ctext in evidence_items:
            print(f"\n[{cid}]\n{ctext}")

        terms = debug["detected_terms"]
        syns = debug["synonyms_used"]
        matchers = build_matchers(syns)

        if not terms:
            print("\nDecision: INSUFFICIENT EVIDENCE (no recognized privacy term in query)")
            print("Tip: Ask using a specific term like location/IMEI/Aadhaar/Advertising ID/etc.")
            print("\nSource:")
            print(f"- {store['app_name']} ({store['url']})")
            continue

        # Deterministic path for mention queries
        if is_mention_query(question):
            if not debug["explicit_found_anywhere"]:
                print("\nDecision: NOT EXPLICITLY MENTIONED (no explicit term match in policy text)")
                print(f"Detected terms: {terms}")
                print("Note: Showing closest semantic chunks above for human review.")
                print("\nSource:")
                print(f"- {store['app_name']} ({store['url']})")
                continue

            print("\nDecision: EXPLICITLY MENTIONED")
            print(f"Detected terms: {terms}")

            explicit_cids = []
            for cid, ctext in evidence_items:
                if chunk_contains_any(ctext, matchers):
                    explicit_cids.append((cid, ctext))

            if not explicit_cids:
                for idx in debug["explicit_global_indices"][:3]:
                    cid = f"{app_slug}#C{idx}"
                    explicit_cids.append((cid, chunks[idx]))

            print("\nMatched sentence(s):")
            for cid, ctext in explicit_cids[:3]:
                hits = extract_matching_sentences(ctext, matchers)
                for h in hits[:3]:
                    print(f'- "{h}" [{cid}]')

            print("\nSource:")
            print(f"- {store['app_name']} ({store['url']})")
            continue

        # Non-mention strict refusal if no explicit evidence anywhere
        if terms and not debug["explicit_found_anywhere"]:
            print("\nDecision: NOT EXPLICITLY MENTIONED (insufficient explicit evidence)")
            print(f"Detected terms: {terms}")
            print("Reason: No chunk in this policy contained the detected term(s). Showing closest semantic chunks above.")
            print("\nSource:")
            print(f"- {store['app_name']} ({store['url']})")
            continue

        # Otherwise: LLM path, but with citation validation
        contexts_labeled = [f"[{cid}] {ctext}" for cid, ctext in evidence_items]
        allowed_ids = {cid for cid, _ in evidence_items}

        answer = ollama_chat(contexts_labeled, question, model=CHAT_MODEL)

        ok, report = validate_llm_answer(answer, allowed_ids)
        if not ok:
            print("\nDecision: REJECTED (invalid/missing citations in model output)")
            print(f"Reason: {report.get('reason')}")
            if report.get("invalid_citations"):
                print(f"Invalid citations: {report['invalid_citations']}")
            if report.get("bullets_missing_valid_citation", 0) > 0:
                print(f"Bullets missing valid citation: {report['bullets_missing_valid_citation']}/{report.get('bullets_checked', 0)}")
            print("Action: Not showing model answer. Use the evidence chunks above.")
            print("\nSource:")
            print(f"- {store['app_name']} ({store['url']})")
            continue

        print("\nAnswer (validated citations):\n", answer)
        print("\nSource:")
        print(f"- {store['app_name']} ({store['url']})")