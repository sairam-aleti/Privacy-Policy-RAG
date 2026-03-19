from typing import List, Dict, Any

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import json
from collections import defaultdict
from core.vector_store import load_app_store, slugify  # type: ignore
from core.verifier import select_evidence, build_matchers, extract_matching_sentences, chunk_contains_any  # type: ignore

def run_experiment(app_slug: str, test_cases: List[Dict[str, Any]], out_file: str):
    print(f"Loading '{app_slug}' store...")
    try:
        chunks, index, meta = load_app_store(app_slug)
    except FileNotFoundError:
        print(f"Store for {app_slug} not found, skipping.")
        return []
    
    results = []
    
    with open(out_file, "a", encoding="utf-8") as f:
        for tc in test_cases:
            target_term = tc.get("data_type", "unknown")
            finding_id = tc.get("finding_id", "UNKNOWN")
            
            print(f"\n--- Testing finding_id: '{finding_id}' | app: {app_slug} | type: '{target_term}' ---")
            
            query = json.dumps(tc)
            
            # Run the selection flow
            evidence_items, debug = select_evidence(chunks, index, app_slug, query)
            terms = debug["detected_terms"]
            
            decision = "UNKNOWN"
            matched_sentences = []
            
            if not terms:
                decision = "INSUFFICIENT EVIDENCE"
            elif not debug["explicit_found_anywhere"]:
                decision = "NOT EXPLICITLY MENTIONED"
            else:
                decision = "EXPLICITLY MENTIONED"
                matchers = build_matchers(debug["synonyms_used"])
                
                explicit_cids = []
                for cid, ctext in evidence_items:
                    if chunk_contains_any(ctext, matchers):
                        explicit_cids.append((cid, ctext))
                
                if not explicit_cids:
                    global_indices = list(debug.get("explicit_global_indices", []))  # type: ignore
                    for idx_val in global_indices[:3]:  # type: ignore
                        try:
                            int_val = int(idx_val)
                            if int_val < len(chunks):
                                cid = f"{app_slug}#C{int_val}"
                                explicit_cids.append((cid, chunks[int_val]))  # type: ignore
                        except (ValueError, TypeError):
                            pass
                        
                for i_match, (cid, ctext) in enumerate(explicit_cids):
                    if i_match >= 3:
                        break
                    if isinstance(ctext, str):
                        hits = extract_matching_sentences(ctext, matchers)  # type: ignore
                        for i_hit, h in enumerate(hits):
                            if i_hit >= 3:
                                break
                            matched_sentences.append({"sentence": h, "citation": cid})
                        
            res = {
                "finding_id": finding_id,
                "app": app_slug,
                "target_term": target_term,
                "query_used": query,
                "detected_terms": terms,
                "decision": decision,
                "matched_sentences": matched_sentences,
                "citations": [cid for cid, _ in evidence_items]
            }
            results.append(res)
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            
            print(f"Decision: {decision}")
            print(f"Detected terms: {terms}")
            if matched_sentences:
                print("Matches:")
                for m in matched_sentences:
                    print(f"  - \"{m['sentence']}\" [{m['citation']}]")
            print(f"Citations Retrieved: {[cid for cid, _ in evidence_items]}")
            
    return results

if __name__ == "__main__":
    fake_data_path = "fake_data.jsonl"
    out_file = "experiment_results.jsonl"
    
    # Empty the output file first
    open(out_file, "w").close()
    
    if not os.path.exists(fake_data_path):
        print(f"Error: {fake_data_path} not found.")
        sys.exit(1)
        
    # Group test cases by app_slug
    test_cases_by_app = defaultdict(list)
    with open(fake_data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tc = json.loads(line)
            app_slug = slugify(tc.get("app_name", "unknown"))
            test_cases_by_app[app_slug].append(tc)
            
    print(f"Loaded {sum(len(v) for v in test_cases_by_app.values())} test cases across {len(test_cases_by_app)} apps.")
    
    for app_slug, cases in test_cases_by_app.items():
        print(f"\n=== Running {app_slug} Experiments ===")
        run_experiment(app_slug, cases, out_file)
        
    print(f"\nExperiments complete. Results saved to {out_file}")
