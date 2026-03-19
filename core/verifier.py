import re
import json
import faiss  # type: ignore
from typing import List, Tuple, Any, Dict, Set
from .llm import ollama_embed, EMBED_MODEL  # type: ignore
from .vector_store import search_index  # type: ignore

PRIVACY_TERMS = {
    "location": ["location", "gps", "geolocation", "precise location", "lat", "lon", "latitude", "longitude"],
    "imei": ["imei"],
    "aadhaar": ["aadhaar", "aadhar"],
    "advertising_id": ["advertising id", "ad id", "aaid", "google advertising id"],
    "android_id": ["android id"],
    "device_id": ["device id", "device identifier", "unique identifier", "hardware id"],
    "mac_address": ["mac address", "mac"],
    "ip_address": ["ip address", "ip"],
    "contacts": ["contacts", "address book", "phonebook", "contact list", "phone number", "email address", "email", "phone"],
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
    "device_info": ["device info", "device information", "mobile device information", "device details"],
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

def extract_query_text(question: str) -> str:
    # If the user provides a structured JSON finding, extract data_type to help with keyword search
    try:
        data = json.loads(question)
        if isinstance(data, dict):
            # We want to match against the data_type if present
            dt = data.get("data_type", "")
            if dt:
                return str(dt)
    except Exception:
        pass
    return question


def detect_terms(question: str) -> List[str]:
    q = extract_query_text(question).lower()
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


def build_matchers(syns: List[str]) -> List[Tuple[str, Any]]:
    matchers: List[Tuple[str, Any]] = []
    for s in syns:
        s = str(s).strip().lower()
        if not s:
            continue
        if (len(s) <= 3 and " " not in s) or s in ["aadhaar", "aadhar"]:
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

_CIT_RE = re.compile(r"\[([^\[\]]+)\]")

def extract_citation_ids(answer_text: str) -> List[str]:
    return _CIT_RE.findall(answer_text or "")


def is_insufficient_evidence_answer(answer_text: str) -> bool:
    if not answer_text:
        return True
    return "insufficient evidence" in answer_text.lower()


def validate_llm_answer(answer_text: str, allowed_ids: Set[str]) -> Tuple[bool, Dict[str, Any]]:
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
    if invalid and len(invalid) == len(all_ids):  # Only fail if ALL citations are completely invalid
        report["reason"] = "all_citations_invalid"
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
        if report["bullets_missing_valid_citation"] == report["bullets_checked"] and report["bullets_checked"] > 0:
            report["reason"] = "all_bullets_missing_valid_citation"
            return False, report

    if not bullet_lines and len(all_ids) == 0:
        report["reason"] = "no_citations_present"
        return False, report

    report["reason"] = "ok"
    return True, report


K_SEMANTIC = 20
K_RETURN = 10
MAX_EXPLICIT = 2

def select_evidence(chunks: List[str], index: faiss.Index, app_slug: str, question: str) -> Tuple[List[Tuple[str, str]], Dict[str, Any]]:
    # Embed the raw JSON string so we don't change how it semantically matches text.
    terms = detect_terms(question)
    syns = synonyms_for(terms)
    matchers = build_matchers(syns)

    query_embedding = ollama_embed([question], model=EMBED_MODEL)[0]
    cand_indices = search_index(index, query_embedding, k=K_SEMANTIC)

    explicit_in_candidates: List[int] = []
    if syns:
        for val in cand_indices:
            try:
                idx = int(val)
                if chunk_contains_any(chunks[idx], matchers):  # type: ignore
                    explicit_in_candidates.append(int(idx))  # type: ignore
            except (ValueError, TypeError):
                pass

    explicit_global: List[int] = []
    if syns:
        for i, ch in enumerate(chunks):
            if chunk_contains_any(ch, matchers):
                explicit_global.append(i)

    chosen = []
    explicit_chosen = []

    for idx in explicit_in_candidates:
        if idx not in chosen:
            chosen.append(int(str(idx)))
            explicit_chosen.append(int(str(idx)))
        if len(explicit_chosen) >= MAX_EXPLICIT:
            break

    if len(explicit_chosen) < MAX_EXPLICIT:
        for idx in explicit_global:
            if idx not in chosen:
                chosen.append(int(str(idx)))
                explicit_chosen.append(int(str(idx)))
            if len(explicit_chosen) >= MAX_EXPLICIT:
                break

    for idx in cand_indices:
        if len(chosen) >= K_RETURN:
            break
        if idx not in chosen:
            chosen.append(int(str(idx)))

    evidence_items = []
    for i, idx_val in enumerate(chosen):
        if i >= K_RETURN:
            break
        try:
            int_idx = int(idx_val)
            cid = f"{app_slug}#C{int_idx}"
            evidence_items.append((cid, chunks[int_idx]))  # type: ignore
        except (ValueError, TypeError):
            continue

    debug = {
        "detected_terms": terms,
        "synonyms_used": syns,
        "explicit_found_anywhere": bool(explicit_global),
        "explicit_global_indices": [explicit_global[i] for i in range(min(50, len(explicit_global)))],  # type: ignore
        "explicit_global_count": len(explicit_global),
        "explicit_selected_count": len(explicit_chosen),
        "semantic_candidate_count": len(cand_indices),
    }
    return evidence_items, debug
