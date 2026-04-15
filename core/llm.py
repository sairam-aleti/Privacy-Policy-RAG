import json
import requests  # type: ignore
from typing import List

OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"

EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3:8b"


def ollama_embed(texts: List[str], model: str = EMBED_MODEL, timeout: int = 120) -> List[List[float]]:
    embeddings = []
    total = len(texts)
    for i, text in enumerate(texts):
        if i % 25 == 0 and i > 0:
            print(f"    ... {i}/{total} chunks embedded")
            
        resp = requests.post(
            OLLAMA_EMBED_URL,
            json={"model": model, "prompt": text},
            timeout=timeout,
        )
        resp.raise_for_status()
        embeddings.append(resp.json()["embedding"])
    return embeddings


def ollama_chat(contexts_labeled: List[str], finding_json: str, model: str = CHAT_MODEL, timeout: int = 300) -> str:
    context = "\n\n".join(contexts_labeled)

    system_msg = (
        "You are a HIGHLY STRICT forensic privacy auditor. Your goal is to detect ANY unauthorized data collection in a bundle.\n\n"
        "Audit Rules:\n"
        "1) Explicit Authorization Only: Tips or security advice do not constitute data collection authorization.\n"
        "2) Zero Tolerance for Undisclosed Data: If an attribute is completely missing from the policy context, it is unauthorized.\n"
        "2a) Safe Harbor & Semantic Deductions: Be highly intelligent. If the policy authorizes 'Personal Data', 'Contact Information', or uses standard business operations (e.g., 'Account Registration'), you MUST deduce that basic functional legacy identifiers (Name, Email, Phone, Address) are authorized. Do NOT fail an attribute if the context inherently covers it semantically.\n"
        "3) Logical Consistency Check: Do NOT contradict yourself. If your analysis states an attribute is authorized or covered by a Safe Harbor, your conclusion MUST reflect that it is authorized.\n"
        "4) Contradiction Detection: Flag if the policy promises privacy but the finding collects invasive tech telemetry.\n\n"
        "Formatting Instructions (MANDATORY):\n"
        "- Separate your analysis into three sections: ### INCIDENT SUMMARY, ### DETAILED ANALYSIS, and ### AUDITOR CONCLUSION.\n"
        "- Under ### INCIDENT SUMMARY, write exactly 1-2 sentences summarizing the finding. You MUST explicitly state the 'collection_context' (e.g., 'during Background Telemetry').\n"
        "- Use double newlines before and after every section header.\n"
        "- Use bullet points (•) for the detailed analysis of each attribute, placing EACH bullet on a NEW line.\n"
        "- DO NOT use markdown bolding (asterisks **). Keep text completely plain.\n"
        "- DO NOT echo internal rules in the output (e.g., avoid writing 'Silence = Violation' or 'Rule 2'). Write professionally and objectively.\n"
        "- Citations must use [chunk_id] format.\n"
        "- At the VERY END of your response, after the conclusion, state the final verdict exactly as: VERDICT: [FOLLOWING], VERDICT: [NOT_FOLLOWING], or VERDICT: [INSUFFICIENT]. Do NOT put the verdict at the top.\n"
    )

    user_msg = (
        f"Policy Context:\n{context}\n\n"
        f"JSON Finding (Bundle to Verify):\n{finding_json}\n"
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

    full_response: str = ""
    for line in resp.iter_lines():
        if not line:
            continue
        try:
            chunk = json.loads(line.decode("utf-8"))
        except Exception:
            continue
        if isinstance(chunk, dict) and "message" in chunk and isinstance(chunk["message"], dict) and "content" in chunk["message"]:
            msg_content = chunk["message"]["content"]
            if isinstance(msg_content, str):
                full_response = str(full_response) + str(msg_content)  # type: ignore
        elif isinstance(chunk, dict) and "content" in chunk:
            content = chunk["content"]
            if isinstance(content, str):
                full_response = str(full_response) + str(content)  # type: ignore

    return str(full_response).strip()


def ollama_verify_chunk(chunk_text: str, finding_json: str, model: str = CHAT_MODEL, timeout: int = 240) -> bool:
    system_msg = (
        "You are a boolean evaluator. "
        "Read the JSON Finding and identify the value of the 'data_type' key (e.g. location, aadhaar). "
        "Then, read the provided Text. "
        "If the Text mentions that specific data type value, output exactly 'YES', even if the destination or purpose differ. "
        "If the specific data type is not mentioned at all, output exactly 'NO'. "
        "Do not provide explanations."
    )
    user_msg = f"JSON Finding:\n{finding_json}\n\nText: {chunk_text}"
    
    try:
        resp = requests.post(
            OLLAMA_CHAT_URL,
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                "stream": False
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        result = resp.json()
        msg = result.get("message", {}).get("content", "").strip().upper()
        return "YES" in msg
    except Exception as e:
        print(f"Verifier API error: {e}")
        return False
