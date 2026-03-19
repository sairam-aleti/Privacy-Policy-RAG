import os
import json
import urllib3  # type: ignore
import uuid
from datetime import datetime
from typing import Dict, Any, List

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from core.vector_store import build_or_load_app_store, read_apps_file, app_key  # type: ignore
from core.verifier import select_evidence, build_matchers, is_mention_query, chunk_contains_any, extract_matching_sentences, validate_llm_answer, normalize_question  # type: ignore
from core.llm import ollama_verify_chunk, ollama_chat, CHAT_MODEL, EMBED_MODEL  # type: ignore


def generate_fake_finding(app_name: str, data_type: str) -> str:
    """Generates a structured JSON finding block for the LLM to verify against."""
    d = {
        "finding_id": f"F-{str(uuid.uuid4())[:8].upper()}",  # type: ignore
        "app_name": app_name,
        "action": "data collection",
        "data_type": data_type,
        "destination": "first party server",
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }
    return json.dumps(d, indent=2)

# ----------------------------
# Main CLI
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
        raw_input_str = input("\nEnter app name, 'list', 'quit', or paste JSON payload directly:\n> ").strip()
        if raw_input_str.lower() == "quit":
            break
        if raw_input_str.lower() == "list":
            print("\nAvailable apps:")
            for info in stores.values():  # type: ignore
                print(f"- {info.get('app_name', 'Unknown') if isinstance(info, dict) else 'Unknown'}")
            continue

        raw_app = raw_input_str
        question_str = ""
        
        # Check if the user pasted a full JSON payload
        try:
            q_data = json.loads(raw_input_str)
            if isinstance(q_data, dict) and "app_name" in q_data:
                raw_app = q_data["app_name"]
                question_str = raw_input_str
        except Exception:
            pass

        k = app_key(raw_app)
        store = {}
        for key, store_info in stores.items():  # type: ignore
            if key == k:
                if isinstance(store_info, dict):
                    store = store_info
                break
                
        if not store:
            print("App not found. Try 'list' and copy the exact name, or enter a close variant.")
            continue

        if not question_str:
            question_str = input("Enter your question/finding (plain text or JSON with 'data_type'): ").strip()
            if question_str.lower() == "quit":
                break
            
        # Parse it to see if it is JSON, otherwise assume it's a data_type description
        try:
            q_data = json.loads(question_str)
            if isinstance(q_data, dict) and "data_type" in q_data:
                question = question_str
            else:
                question = normalize_question(question_str)
        except Exception:
            # If plain English, wrap it in our structured prompt
            question = generate_fake_finding(str(store.get('app_name', 'Unknown')), question_str)

        app_slug: str = str(store.get("app_slug", ""))  # type: ignore
        chunks: List[str] = store.get("chunks", [])  # type: ignore
        index: Any = store.get("index")  # type: ignore

        # Provide `question_str` correctly to FAISS search - actually wait, embedding the structured JSON might hurt semantic search. 
        # But we wrote logic to embed the raw question string in vector_store. We passed `question` directly.
        
        evidence_items_raw, debug_raw = select_evidence(chunks, index, app_slug, question)  # type: ignore
        evidence_items = list(evidence_items_raw) if isinstance(evidence_items_raw, list) else []
        debug = dict(debug_raw) if isinstance(debug_raw, dict) else {}  # type: ignore

        print("\nClosest evidence (top 10 chunks):")
        for cid, ctext in evidence_items[:10]:  # type: ignore
            print(f"\n[{cid}]\n{ctext}")

        terms = debug["detected_terms"]
        syns = debug["synonyms_used"]
        matchers = build_matchers(syns)

        if not terms:
            print("\nDecision: INSUFFICIENT EVIDENCE (no recognized privacy term in query)")
            print("Tip: Ask using a specific term like location/IMEI/Aadhaar/Advertising ID/etc.")
            print("\nSource:")
            print(f"- {store.get('app_name', 'Unknown')} ({store.get('url', 'Unknown')})")
            continue

        # Deterministic path for mention queries
        if is_mention_query(question):
            if debug.get("explicit_found_anywhere") is not True:
                print("\nDecision: NOT EXPLICITLY MENTIONED (no explicit term match in policy text)")
                print(f"Detected terms: {terms}")
                print("Note: Showing closest semantic chunks above for human review.")
                print("\nSource:")
                print(f"- {store.get('app_name', 'Unknown')} ({store.get('url', 'Unknown')})")
                continue

            print("\nDecision: EXPLICITLY MENTIONED")
            print(f"Detected terms: {terms}")

            explicit_cids = []
            for cid, ctext in evidence_items:
                if chunk_contains_any(ctext, matchers):
                    explicit_cids.append((cid, ctext))

            if not explicit_cids:
                global_indices = list(debug.get("explicit_global_indices", []))  # type: ignore
                for val in global_indices[:3]:  # type: ignore
                    try:
                        int_val = int(val)
                        if int_val < len(chunks):
                            cid_global = f"{app_slug}#C{int_val}"
                            explicit_cids.append((cid_global, str(chunks[int_val])))  # type: ignore
                    except (ValueError, TypeError):
                        pass

            print("\nMatched sentence(s):")
            for cid, ctext in explicit_cids[:3]:  # type: ignore
                if isinstance(ctext, str):
                    hits = extract_matching_sentences(ctext, matchers)  # type: ignore
                    for h in hits[:3]:  # type: ignore
                        print(f'- "{h}" [{cid}]')

            print("\nSource:")
            print(f"- {store.get('app_name', 'Unknown')} ({store.get('url', 'Unknown')})")  # type: ignore
            continue

        # Non-mention strict refusal if no explicit evidence anywhere
        if terms and not debug["explicit_found_anywhere"]:
            print("\nDecision: NOT EXPLICITLY MENTIONED (insufficient explicit evidence)")
            print(f"Detected terms: {terms}")
            print("Reason: No chunk in this policy contained the detected term(s). Showing closest semantic chunks above.")
            print("\nSource:")
            print(f"- {store.get('app_name', 'Unknown')} ({store.get('url', 'Unknown')})")  # type: ignore
            continue

        # Otherwise: LLM path, but with verified chunks + citation validation
        print("\nVerifying retrieved chunks dynamically with LLM...")
        verified_evidence = []
        K_RETURN = 10
        for cid, ctext in evidence_items[:K_RETURN]:  # type: ignore
            if ollama_verify_chunk(ctext, question, model=CHAT_MODEL):
                verified_evidence.append((cid, ctext))
                
        if not verified_evidence:
            print("\nDecision: REJECTED (Verifier discarded all chunks as irrelevant)")
            print("Action: Not showing model answer. Try rephrasing the question.")
            print("\nSource:")
            print(f"- {store.get('app_name', 'Unknown')} ({store.get('url', 'Unknown')})")  # type: ignore
            continue
            
        print(f"\nVerifier passed {len(verified_evidence)} chunks out of {K_RETURN}:")
        for cid, _ in verified_evidence:
            print(f"  ✓ Accepted: {cid}")
        
        # Keep up to top 5 verified
        final_evidence = verified_evidence[:5] if len(verified_evidence) > 5 else verified_evidence  # type: ignore
        contexts_labeled = [f"[{cid}] {ctext}" for cid, ctext in final_evidence]
        allowed_ids = {cid for cid, _ in final_evidence}

        answer = ollama_chat(contexts_labeled, question, model=CHAT_MODEL)

        ok, report = validate_llm_answer(answer, allowed_ids)
        if not ok:
            print("\nDecision: REJECTED (invalid/missing citations in model output)")
            print(f"Reason: {report.get('reason')}")  # type: ignore
            if report.get("invalid_citations"):  # type: ignore
                print(f"Invalid citations: {report.get('invalid_citations')}")  # type: ignore
            if report.get("bullets_missing_valid_citation", 0) > 0:  # type: ignore
                print(f"Bullets missing valid citation: {report.get('bullets_missing_valid_citation', 0)}/{report.get('bullets_checked', 0)}")  # type: ignore
            print("Action: Not showing model answer. Use the evidence chunks above.")
            print("\nSource:")
            print(f"- {store.get('app_name', 'Unknown')} ({store.get('url', 'Unknown')})")  # type: ignore
            continue

        print("\nAnswer (validated citations):\n", answer)
        print("\nSource:")
        print(f"- {store.get('app_name', 'Unknown')} ({store.get('url', 'Unknown')})")  # type: ignore