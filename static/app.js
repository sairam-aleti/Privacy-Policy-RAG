/* ============================================
   Privacy Policy RAG — Frontend Logic
   ============================================ */

document.addEventListener("DOMContentLoaded", () => {
    const dropzone = document.getElementById("dropzone");
    const fileInput = document.getElementById("fileInput");
    const singleInput = document.getElementById("singleInput");
    const singleBtn = document.getElementById("singleBtn");
    const progressSection = document.getElementById("progressSection");
    const progressTitle = document.getElementById("progressTitle");
    const progressCount = document.getElementById("progressCount");
    const progressFill = document.getElementById("progressFill");
    const resultsSection = document.getElementById("resultsSection");
    const resultsGrid = document.getElementById("resultsGrid");
    const resultsStats = document.getElementById("resultsStats");
    const exportBtn = document.getElementById("exportBtn");
    const ollamaStatus = document.getElementById("ollamaStatus");
    const appCountEl = document.getElementById("appCount");

    let allResults = [];

    // ── Check API status ──
    async function checkStatus() {
        try {
            const resp = await fetch("/api/apps");
            const data = await resp.json();
            const dot = ollamaStatus.querySelector(".status-dot");
            const text = ollamaStatus.querySelector(".status-text");
            dot.classList.add("online");
            text.textContent = "Online";
            appCountEl.textContent = `${data.apps.length} apps`;
        } catch {
            const text = ollamaStatus.querySelector(".status-text");
            text.textContent = "Offline";
        }
    }
    checkStatus();

    // ── Dropzone events ──
    dropzone.addEventListener("click", () => fileInput.click());

    dropzone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropzone.classList.add("dragover");
    });

    dropzone.addEventListener("dragleave", () => {
        dropzone.classList.remove("dragover");
    });

    dropzone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropzone.classList.remove("dragover");
        const files = e.dataTransfer.files;
        if (files.length > 0) handleFile(files[0]);
    });

    fileInput.addEventListener("change", () => {
        if (fileInput.files.length > 0) handleFile(fileInput.files[0]);
    });

    // ── Single finding ──
    singleBtn.addEventListener("click", () => {
        const text = singleInput.value.trim();
        if (!text) return;
        try {
            const finding = JSON.parse(text);
            processBatch([finding]);
        } catch {
            alert("Invalid JSON. Please paste a valid JSON finding.");
        }
    });

    // ── File handler ──
    function handleFile(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            const lines = e.target.result.split("\n").filter(ln => ln.trim());
            const findings = [];
            for (const ln of lines) {
                try { findings.push(JSON.parse(ln)); } catch { /* skip */ }
            }
            if (findings.length === 0) {
                alert("No valid JSONL findings found in file.");
                return;
            }
            processBatch(findings);
        };
        reader.readAsText(file);
    }

    // ── Batch processor  ──
    async function processBatch(findings) {
        allResults = [];
        resultsGrid.innerHTML = "";
        resultsSection.classList.remove("hidden");
        progressSection.classList.remove("hidden");
        singleBtn.disabled = true;

        const total = findings.length;
        progressTitle.textContent = `Processing ${total} finding${total > 1 ? 's' : ''}…`;

        for (let i = 0; i < total; i++) {
            progressCount.textContent = `${i + 1} / ${total}`;
            progressFill.style.width = `${((i + 1) / total) * 100}%`;

            try {
                const resp = await fetch("/api/verify-single", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(findings[i]),
                });
                const result = await resp.json();
                allResults.push(result);
                renderCard(result, i);
            } catch (err) {
                const errorResult = {
                    finding_id: findings[i].finding_id || `F-${i}`,
                    app_name: findings[i].app_name || "Unknown",
                    data_type: findings[i].data_type || "unknown",
                    status: "ERROR",
                    error: err.message,
                    answer: "",
                    evidence: [],
                };
                allResults.push(errorResult);
                renderCard(errorResult, i);
            }
        }

        progressTitle.textContent = "Processing complete!";
        singleBtn.disabled = false;
        updateStats();
    }

    // ── Render a result card ──
    function renderCard(result, index) {
        const card = document.createElement("div");
        card.className = "result-card";
        card.style.animationDelay = `${index * 0.08}s`;

        const statusClass = `badge-${result.status.toLowerCase()}`;
        const statusLabel = result.status.replace(/_/g, " ");

        // Tags
        const tags = [];
        if (result.action) tags.push(result.action);
        if (result.destination) tags.push(result.destination);
        if (result.purpose) tags.push(result.purpose);
        const tagsHtml = tags.map(t => `<span class="tag">${escapeHtml(t)}</span>`).join("");

        // Answer
        let answerHtml = "";
        if (result.answer) {
            answerHtml = `<div class="answer-text">${formatAnswer(result.answer)}</div>`;
        }

        // Error
        let errorHtml = "";
        if (result.error && result.status !== "VERIFIED") {
            errorHtml = `<div class="error-msg">${escapeHtml(result.error)}</div>`;
        }

        // Verification stats
        let statsHtml = "";
        if (result.verified_chunks !== undefined && result.total_chunks_checked) {
            statsHtml = `<span class="tag" style="color: var(--amber);">⚡ Hybrid RRF Search</span> <span class="tag">✓ Top ${result.verified_chunks} context chunks analyzed</span>`;
        }

        // Evidence
        let evidenceHtml = "";
        if (result.evidence && result.evidence.length > 0) {
            const items = result.evidence.map(ev =>
                `<div class="evidence-item">
                    <span class="chunk-id">${escapeHtml(ev.chunk_id || ev.citation || "")}</span>
                    ${escapeHtml(ev.text || ev.sentence || "")}
                </div>`
            ).join("");
            evidenceHtml = `
                <button class="evidence-toggle" onclick="this.nextElementSibling.classList.toggle('open')">
                    ▸ View the AI's sourced evidence (${result.evidence.length} citations)
                </button>
                <div class="evidence-panel">${items}</div>
            `;
        }

        card.innerHTML = `
            <div class="card-header">
                <div class="card-meta">
                    <span class="finding-id">${escapeHtml(result.finding_id || "")}</span>
                    <span class="card-title">
                        <span class="app-name">${escapeHtml(result.app_name || "")}</span>
                        <span class="separator">→</span>
                        <span class="data-type">${escapeHtml(result.data_type || "")}</span>
                    </span>
                </div>
                <span class="status-badge ${statusClass}">${statusLabel}</span>
            </div>
            <div class="card-body">
                <div class="card-tags">${tagsHtml}${statsHtml}</div>
                ${answerHtml}
                ${errorHtml}
            </div>
            ${evidenceHtml}
        `;

        resultsGrid.appendChild(card);
    }

    // ── Update stats bar ──
    function updateStats() {
        const counts = { VERIFIED: 0, PARTIALLY_VERIFIED: 0, REJECTED: 0, NOT_FOUND: 0, INSUFFICIENT: 0, ERROR: 0 };
        for (const r of allResults) {
            const s = r.status || "ERROR";
            if (s in counts) counts[s]++;
            else counts.ERROR++;
        }
        const verified = counts.VERIFIED + counts.PARTIALLY_VERIFIED;
        const rejected = counts.REJECTED + counts.ERROR;
        const insufficient = counts.NOT_FOUND + counts.INSUFFICIENT;

        resultsStats.innerHTML = `
            <span class="stat-badge stat-verified">✓ ${verified} Verified</span>
            <span class="stat-badge stat-rejected">✗ ${rejected} Rejected</span>
            <span class="stat-badge stat-insufficient">⚠ ${insufficient} Insufficient</span>
        `;
    }

    // ── Export ──
    exportBtn.addEventListener("click", () => {
        const blob = new Blob([JSON.stringify(allResults, null, 2)], { type: "application/json" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `rag_results_${new Date().toISOString().slice(0, 10)}.json`;
        a.click();
        URL.revokeObjectURL(url);
    });

    // ── Helpers ──
    function escapeHtml(str) {
        const div = document.createElement("div");
        div.textContent = str || "";
        return div.innerHTML;
    }

    function formatAnswer(text) {
        // Convert markdown bullets to styled html
        return escapeHtml(text)
            .replace(/^\* /gm, "• ")
            .replace(/^- /gm, "• ")
            .replace(/\[([^\]]+)\]/g, '<strong>[$1]</strong>');
    }
});
