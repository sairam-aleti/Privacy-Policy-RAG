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
        "You are a HIGHLY STRICT privacy auditor. Your goal is to detect unauthorized data collection.\n"
        "Your response MUST start with a clear Verdict on its own line: [FOLLOWING], [NOT_FOLLOWING], or [INSUFFICIENT].\n"
        "Audit Rules:\n"
        "1) USE ONLY the provided Context. Silence does NOT equal permission.\n"
        "2) If the Context does not EXPLICITLY state that the data_type can be collected for the specific purpose/context, you MUST return [NOT_FOLLOWING].\n"
        "3) If the Context contains a 'Privacy Promise' or policy (e.g., 'We do not sell data') and the Finding performs that prohibited action, you MUST return [NOT_FOLLOWING].\n"
        "4) Read the 'data_type' and 'collection_context' carefully. If the user is a minor and the policy forbids under-18 collection, return [NOT_FOLLOWING].\n"
        "5) Explanations must be 2-5 bullets with [chunk_id] citations. Do NOT cite the finding_id.\n"
        "6) Be pedantic. If you are 1% unsure, return [INSUFFICIENT] or [NOT_FOLLOWING].\n"
    )

    user_msg = (
        f"Context:\n{context}\n\n"
        f"JSON Finding to verify:\n{finding_json}\n"
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
