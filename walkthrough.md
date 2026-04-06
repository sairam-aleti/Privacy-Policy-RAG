# 🕵️ Privacy Policy RAG — Cross-App Exposure Tracker

The dashboard has been upgraded from a generic compliance auditor into a professional identity vulnerability tracker. It can now correlate unauthorized data collection across multiple applications for specific user profiles.

## Key Upgrades

### 1. Cross-App Exposure Tracker
We added a professional "Cross-App Exposure Tracker" module. 
- **Purpose**: Correlate unauthorized data collection targeting a specific user profile across the verified network.
- **Logic**: It identifies "collected" PII (Full Name, Mobile Number, Permanent Address) within findings that have a `REJECTED` or `NOT_FOUND` status.
- **UI**: A dedicated search interface allows tracing exposures by specific PII attributes.

### 2. Professional Data Schema
- Migrated from "leaked" terminology to professional `collected_name`, `collected_phone`, and `collected_address` keys.
- Updated [fake_data.jsonl](file:///home/apf/Privacy-Policy-RAG/fake_data.jsonl) with 3 persistent "victim" profiles scattered across 20 test cases to simulate real-world cross-app tracking.

### 3. Frozen Vector Stores (No-Update Mode)
- **Auto-Update Disabled**: Modified [core/vector_store.py](file:///home/apf/Privacy-Policy-RAG/core/vector_store.py) to prevent the system from automatically scraping websites or re-embedding chunks on startup.
- **Local Priority**: The system now strictly uses the pre-built embeddings in the `stores/` directory.

### 4. Logic & Model Refinement
- **Semantic Path Fix**: Removed a legacy "fast-fail" string check in `server.py` that was blocking the AI from finding semantic matches (e.g., matching "whereabouts" to "location").
- **8B Parameter Reasoning**: The backend is powered by `llama3:8b`, providing high-recall auditing of complex legal contexts.

---

## How to Run

```bash
# 1. Ensure Ollama is running with the required models
ollama run llama3:8b
ollama run nomic-embed-text

# 2. Activate environment and start server
source venv/bin/activate
python server.py

# 3. Access Dashboard
# → http://localhost:5000
```

## Testing the Tracker

1. **Upload**: Drag and drop [fake_data.jsonl](file:///home/apf/Privacy-Policy-RAG/fake_data.jsonl) into the dashboard.
2. **Audit**: Wait for the 8B model to finish the batch verification.
3. **Trace**: In the **Cross-App Exposure Tracker** box, enter a known test number like `+91-9876500001`.
4. **Expose**: Click **Trace Exposures** to see a grouped summary of every app that is currently handling that specific user's data without policy authorization.

---

## Verification Summary

- [x] Terminology updated to "collected_*" across Backend/Frontend/Data.
- [x] Auto-scraping/Auto-embedding disabled in `vector_store.py`.
- [x] Professional UI implemented (no emojis, clean amber/espresso glassmorphism).
- [x] Cross-app correlation logic verified in `app.js`.
