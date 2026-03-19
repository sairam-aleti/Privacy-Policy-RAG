import json
import requests  # type: ignore
from typing import List

OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"

EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3.2:3b"


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
        "You are a strict privacy-policy auditor verifying a specific finding.\n"
        "Rules:\n"
        "1) Use ONLY the provided Context. Do NOT use outside knowledge.\n"
        "2) Read the provided JSON Finding. Report what the Context says regarding the requested data and action.\n"
        "3) Return 2-6 bullet points explaining the evidence found. Each bullet MUST end with one or more citations to chunk IDs exactly as they appear in the Context.\n"
        "Example bullet point: Location data is retrieved and collected [paytm#C11].\n"
        "4) If the Context does not mention the requested data at all, say 'Insufficient evidence in provided context.'\n"
        "5) Do NOT claim 'X not found in policy'. Only speak about what is present in Context.\n"
        "6) Do NOT invent citation IDs. NEVER use the 'finding_id' as a citation.\n"
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


def ollama_verify_chunk(chunk_text: str, finding_json: str, model: str = CHAT_MODEL, timeout: int = 60) -> bool:
    system_msg = (
        "You are a boolean evaluator. "
        "Read the provided Text and the JSON Finding. "
        "Decide if the Text mentions the requested data referenced in the JSON Finding. "
        "If the specific data type is mentioned in the Text, output exactly 'YES', even if the destination or purpose differ. "
        "If the data type is not mentioned at all, output exactly 'NO'. "
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
