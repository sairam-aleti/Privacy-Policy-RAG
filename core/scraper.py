import fitz  # type: ignore
import requests  # type: ignore
from bs4 import BeautifulSoup  # type: ignore
from typing import Tuple

def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    out = []
    for page in doc:
        out.append(page.get_text())
    return "\n".join(out)


def extract_text_from_website(url: str, timeout: int = 30) -> Tuple[str, str]:
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1"
        }
        resp = requests.get(url, timeout=timeout, headers=headers, verify=False)
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
