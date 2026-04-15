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
    """Verify a single finding using the RAG pipeline."""
    app_name = finding.get("app_name", "")
    finding_id = finding.get("finding_id", "")
    action = finding.get("action", "")
    destination = finding.get("destination", "")
    purpose = finding.get("purpose", "")

    # Extract all collected attributes to form the Bundle Query
    collected_fields = {k.replace("collected_", "").replace("_", " ").title(): v 
                        for k, v in finding.items() if k.startswith("collected_") and v}
    
    # Bundle Query Focus
    data_type = ", ".join(collected_fields.keys()) if collected_fields else "General Telemetry"

    result: Dict[str, Any] = {
        "finding_id": finding_id,
        "app_name": app_name,
        "data_type": data_type, # Now represents the entire bundle
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
    # Add all collected fields to the result for UI transparency
    for k, v in finding.items():
        if k.startswith("collected_"):
            result[k] = v

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

    # Construct forensic search question: "Policy regarding [Attributes] during [Context]"
    context = finding.get("collection_context", "Background technical telemetry")
    question = f"What is the privacy policy regarding the collection of {data_type} during {context}?"
    
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



    # All findings must be processed sequentially by the Strict Auditor LLM. 
    # Bypasses such as deterministic BM25 matches have been removed to ensure rigorous bundle-aware logic and proper dossier formatting.

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

    # Filter evidence for UI display to only those explicitly cited by the AI
    cited_ids = set(report.get("all_bracket_ids", []))
    visual_evidence = []
    for cid, ctext in final_evidence:
        if cid in cited_ids:
            visual_evidence.append({"chunk_id": cid, "text": ctext[:500]}) # type: ignore
            
    # If no valid citations were made, just show the top chunk used as fallback
    if not visual_evidence and final_evidence:
        cid, ctext = final_evidence[0]
        visual_evidence.append({"chunk_id": cid, "text": str(ctext)[:500]})

    result["evidence"] = visual_evidence

    answer_upper = answer.upper()
    if "NOT_FOLLOWING" in answer_upper or "NOT FOLLOWING" in answer_upper:
        result["status"] = "REJECTED"
    elif "INSUFFICIENT" in answer_upper:
        result["status"] = "INSUFFICIENT"
    elif "FOLLOWING" in answer_upper:
        result["status"] = "VERIFIED"
    else:
        result["status"] = "INSUFFICIENT"
    
    result["answer"] = answer
    if not ok:
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


@app.route('/api/load-research-data/<app_name>', methods=['GET'])
def load_research_data(app_name):
    try:
        filename = f"fake_data/{app_name.lower().replace(' ', '_')}.jsonl"
        if not os.path.exists(filename):
            # Try to find closely named file
            return jsonify({"error": f"No data found for {app_name}"}), 404
            
        findings = []
        with open(filename, 'r') as f:
            for line in f:
                if line.strip():
                    findings.append(json.loads(line))
        return jsonify({"findings": findings})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/process', methods=['POST'])
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
