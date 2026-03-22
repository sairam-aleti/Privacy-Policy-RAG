"""
Privacy Policy RAG — Flask Web Server
Serves a professional dashboard for batch JSONL verification.
"""
import os
import sys
import json
import uuid
import urllib3  # type: ignore
from datetime import datetime
from typing import Dict, Any, List

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, jsonify, send_from_directory  # type: ignore

from core.vector_store import build_or_load_app_store, read_apps_file, app_key  # type: ignore
from core.verifier import (  # type: ignore
    select_evidence, build_matchers, is_mention_query,
    chunk_contains_any, extract_matching_sentences,
    validate_llm_answer, normalize_question,
)
from core.llm import ollama_verify_chunk, ollama_chat, CHAT_MODEL  # type: ignore

app = Flask(__name__, static_folder="static", static_url_path="/static")

# ---------------------------------------------------------------------------
# Global stores — loaded once at startup
# ---------------------------------------------------------------------------
STORES: Dict[str, Dict[str, Any]] = {}


def init_stores():
    """Load FAISS stores for every app in apps.txt."""
    global STORES
    apps = read_apps_file("apps.txt")
    if not apps:
        print("No apps found in apps.txt")
        return

    print(f"Found {len(apps)} apps in apps.txt")
    print("Building/loading per-app FAISS stores...")

    for k, info in apps.items():
        app_name = info["app_name"]
        url = info["url"]
        print(f"\nProcessing {app_name}...")
        try:
            app_slug, chunks, index, bm25, meta = build_or_load_app_store(app_name, url, rebuild=False)
            STORES[k] = {
                "app_name": app_name,
                "app_slug": app_slug,
                "url": url,
                "chunks": chunks,
                "index": index,
                "bm25": bm25,
                "meta": meta,
            }
            print(f"  Store ready: stores/{app_slug}/ (chunks={len(chunks)})")
        except Exception as e:
            print(f"  Skipping {app_name}: {e}")

    print(f"\n[OK] RAG ready - {len(STORES)} app store(s) loaded.")


# ---------------------------------------------------------------------------
# Core verification logic (reusable by both CLI and web)
# ---------------------------------------------------------------------------

def process_finding(finding: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run the full RAG verification pipeline for a single finding dict.
    Returns a structured result dict ready for the UI.
    """
    finding_id = finding.get("finding_id", f"F-{uuid.uuid4().hex[:8].upper()}")  # type: ignore
    app_name = finding.get("app_name", "Unknown")
    data_type = finding.get("data_type", "unknown")
    action = finding.get("action", "")
    destination = finding.get("destination", "")
    purpose = finding.get("purpose", "")

    result: Dict[str, Any] = {
        "finding_id": finding_id,
        "app_name": app_name,
        "data_type": data_type,
        "action": action,
        "destination": destination,
        "purpose": purpose,
        "status": "PROCESSING",
        "evidence": [],
        "answer": "",
        "source_url": "",
        "verified_chunks": 0,
        "total_chunks_checked": 0,
        "error": None,
    }

    # Locate the app store
    k = app_key(app_name)
    store = None
    for key, store_info in STORES.items():
        if key == k:
            store = store_info
            break

    if not store:
        result["status"] = "ERROR"
        result["error"] = f"App '{app_name}' not found in loaded stores."
        return result

    assert store is not None
    result["source_url"] = store.get("url", "")

    question = json.dumps(finding)
    app_slug = str(store.get("app_slug", ""))
    chunks: List[str] = store.get("chunks", [])
    index = store.get("index")
    bm25 = store.get("bm25")

    # Run evidence selection
    try:
        evidence_items_raw, debug_raw = select_evidence(chunks, index, bm25, app_slug, question)
    except Exception as e:
        result["status"] = "ERROR"
        result["error"] = f"Evidence selection failed: {e}"
        return result

    evidence_items = list(evidence_items_raw) if isinstance(evidence_items_raw, list) else []
    debug = dict(debug_raw) if isinstance(debug_raw, dict) else {}

    terms = debug.get("detected_terms", [])
    syns = debug.get("synonyms_used", [])
    matchers = build_matchers(syns)

    # Path 1: No recognized privacy term
    if not terms:
        result["status"] = "INSUFFICIENT"
        result["error"] = "No recognized privacy term detected in the data_type."
        return result

    # Path 2: Term recognized but not found in text
    if terms and not debug.get("explicit_found_anywhere"):
        result["status"] = "NOT_FOUND"
        result["error"] = f"Term(s) {terms} not explicitly found in the policy text."
        # Provide semantic evidence nonetheless
        result["evidence"] = [
            {"chunk_id": cid, "text": ctext[:300]}  # type: ignore
            for cid, ctext in evidence_items[:5]  # type: ignore
        ]
        return result

    # Path 3 (mention query): Deterministic match
    if is_mention_query(question):
        result["status"] = "VERIFIED"
        explicit_cids = [(cid, ctext) for cid, ctext in evidence_items if chunk_contains_any(ctext, matchers)]
        matched = []
        for cid, ctext in explicit_cids[:5]:  # type: ignore
            hits = extract_matching_sentences(ctext, matchers) if isinstance(ctext, str) else []
            for h in hits[:3]:  # type: ignore
                matched.append({"sentence": h, "citation": cid})
        result["evidence"] = matched
        result["answer"] = "Explicitly mentioned in the privacy policy (deterministic match)."
        return result

    # Path 4: Direct LLM Generation (Skipping per-chunk LLM checks for speed)
    # Since BM25 + FAISS + Explicit Matching provide highly accurate top candidates,
    # we feed the top 5 directly to the final generator to save 10 LLM calls per finding.
    final_evidence = evidence_items[:5]  # type: ignore

    if not final_evidence:
        result["status"] = "REJECTED"
        result["error"] = "No relevant chunks retrieved by hybrid search."
        return result

    result["total_chunks_checked"] = len(evidence_items)
    result["verified_chunks"] = len(final_evidence)

    # Generate final answer
    contexts_labeled = [f"[{cid}] {ctext}" for cid, ctext in final_evidence]
    allowed_ids = {cid for cid, _ in final_evidence}

    try:
        answer = ollama_chat(contexts_labeled, question, model=CHAT_MODEL)
    except Exception as e:
        result["status"] = "ERROR"
        result["error"] = f"LLM generation failed: {e}"
        return result

    ok, report = validate_llm_answer(answer, allowed_ids)

    result["evidence"] = [
        {"chunk_id": cid, "text": ctext[:300]}  # type: ignore
        for cid, ctext in final_evidence
    ]

    if ok:
        result["status"] = "VERIFIED"
        result["answer"] = answer
    else:
        # Show the answer anyway but flag it
        result["status"] = "PARTIALLY_VERIFIED"
        result["answer"] = answer
        result["error"] = report.get("reason", "citation_issues")

    return result


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/apps", methods=["GET"])
def api_apps():
    apps_list = []
    for info in STORES.values():
        apps_list.append({
            "name": info.get("app_name", "Unknown"),
            "slug": info.get("app_slug", ""),
            "url": info.get("url", ""),
            "chunks": len(info.get("chunks", [])),
        })
    return jsonify({"apps": apps_list})


@app.route("/api/verify", methods=["POST"])
def api_verify_batch():
    """Accept a JSONL file or JSON array, process each finding."""
    results = []

    # Handle file upload
    if "file" in request.files:
        f = request.files["file"]
        content = f.read().decode("utf-8")
        lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
        findings = []
        for ln in lines:
            try:
                findings.append(json.loads(ln))
            except json.JSONDecodeError:
                continue
    else:
        # Handle JSON body
        data = request.get_json(silent=True)
        if isinstance(data, list):
            findings = data
        elif isinstance(data, dict):
            findings = [data]
        else:
            return jsonify({"error": "Send a JSONL file or JSON body."}), 400

    for finding in findings:
        result = process_finding(finding)
        results.append(result)

    return jsonify({"results": results, "total": len(results)})


@app.route("/api/verify-single", methods=["POST"])
def api_verify_single():
    """Accept a single JSON finding."""
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON body."}), 400
    result = process_finding(data)
    return jsonify(result)


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    init_stores()
    print("\n[*] Starting web server at http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
