# Privacy Policy RAG Dashboard — Run Guide

This dashboard performs research-aligned forensic auditing of mobile application data collection practices using a locally hosted RAG (Retrieval-Augmented Generation) pipeline.

## 🚀 Step 1: Activate Virtual Environment & LLM Infrastructure
Before running any commands, ensure your environment is active and Ollama is ready.

1.  **Activate Environment**:
    ```bash
    source venv/bin/activate
    ```
2.  **Start Ollama Server**:
    ```bash
    ollama serve
    ```
3.  **Pull Required Models**:
    ```bash
    ollama pull llama3:8b
    ollama pull nomic-embed-text
    ```

## 📂 Step 2: Initialize Digital Footprint (Developer Only)
If you want to regenerate the research-aligned fake data (50 rows per app with technical leaks), run:
```bash
# This script creates/updates the fake_data/ directory
python3 -c "import json; ...script content..." 
```

## ⚙️ Step 3: Launch the Audit Server
Start the Flask backend which manages the FAISS vector stores and the audit pipeline:
```bash
python3 server.py
```
*Note: The server will take a moment to load the 15+ per-app vector stores into memory.*

## 💻 Step 4: Access the Dashboard
1.  Open your browser to: `http://localhost:5000`
2.  **To Audit an App**:
    *   Select an app (e.g., **PayTM** or **Binance**) from the "Cross-App Exposure Tracker" section.
    *   Choose an identifier (e.g., **PAN**, **Email**, or **Phone**).
    *   Enter a value (e.g., `ABCDE1234F` for PAN or `+91-9876500001` for Phone).
    *   Click **Trace Exposures**.
3.  **To Upload Custom Findings**:
    *   Drag and drop any `.jsonl` file into the upload zone at the top.

## 🕵️ Key Forensic Features
*   **Recursive Linkage**: The system automatically hops from PII (like a Phone number) to technical bridges (like Device ID) to uncover "Silent" background tracking that is normally hidden.
*   **Bundle Auditing**: The AI audits the **entire collection bundle** (IP, Battery, Device Info) against the policy. If even one technical attribute is unauthorized, the entire finding is flagged.
*   **Audit Caching**: Results are cached during your session. Switching between apps or re-auditing previously seen records is now instantaneous.

---
*For research purposes only. Analysis is based on the provided privacy policy vector stores.*