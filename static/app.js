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
        let statusLabel = result.status.replace(/_/g, " ");
        
        if (result.status === "VERIFIED" || result.status === "PARTIALLY_VERIFIED") {
            statusLabel = "Following Privacy Policy";
        } else if (result.status === "REJECTED" || result.status === "NOT_FOUND" || result.status === "ERROR") {
            statusLabel = "Not Following";
        } else if (result.status === "INSUFFICIENT") {
            statusLabel = "Insufficient Information";
        }

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
        const following = counts.VERIFIED + counts.PARTIALLY_VERIFIED;
        const notFollowing = counts.REJECTED + counts.NOT_FOUND + counts.ERROR;
        const insufficient = counts.INSUFFICIENT;

        resultsStats.innerHTML = `
            <span class="stat-badge stat-verified">✓ ${following} Following Privacy Policy</span>
            <span class="stat-badge stat-rejected">✗ ${notFollowing} Not Following</span>
            <span class="stat-badge stat-insufficient">⚠ ${insufficient} Insufficient Information</span>
        `;
    }

    // ── Identity Tracker ──
    const traceBtn = document.getElementById("traceBtn");
    const resetTraceBtn = document.getElementById("resetTraceBtn");
    const traceName = document.getElementById("traceName");
    const tracePhone = document.getElementById("tracePhone");
    const traceAddress = document.getElementById("traceAddress");
    const trackerSummary = document.getElementById("trackerSummary");

    function renderFiltered() {
        if (!traceBtn) return; // Prevent errors if DOM missing
        const nameQuery = traceName.value.trim().toLowerCase();
        const phoneQuery = tracePhone.value.trim().toLowerCase();
        const addrQuery = traceAddress.value.trim().toLowerCase();
        
        resultsGrid.innerHTML = "";
        trackerSummary.classList.add("hidden");

        const isFiltering = nameQuery || phoneQuery || addrQuery;
        let traceCount = 0;
        let appsExposed = new Set();

        allResults.forEach((res, idx) => {
            if (!isFiltering) {
                renderCard(res, idx);
                return;
            }

            // Target only data handled without policy alignment (Not Following)
            const isUnsafe = (res.status === "REJECTED" || res.status === "NOT_FOUND" || res.status === "ERROR");
            if (!isUnsafe) return;

            const n = (res.collected_name || "").toLowerCase();
            const p = (res.collected_phone || "").toLowerCase();
            const a = (res.collected_address || "").toLowerCase();

            let match = false;
            // Cross-reference any provided PII
            if (nameQuery && n.includes(nameQuery)) match = true;
            if (phoneQuery && p.includes(phoneQuery)) match = true;
            if (addrQuery && a.includes(addrQuery)) match = true;

            if (match) {
                renderCard(res, idx);
                traceCount++;
                appsExposed.add(res.app_name);
            }
        });

        if (isFiltering) {
            trackerSummary.classList.remove("hidden");
            trackerSummary.innerHTML = `Found <strong>${traceCount} unauthorized privacy leaks</strong> targeting this identity across <strong>${appsExposed.size} separate apps</strong>.`;
            trackerSummary.style.color = "var(--text-primary)";
            trackerSummary.style.background = "var(--red-bg)";
            trackerSummary.style.border = "1px solid rgba(239, 68, 68, 0.4)";
            trackerSummary.style.padding = "12px 18px";
            trackerSummary.style.borderRadius = "6px";
        }
    }

    if (traceBtn) traceBtn.addEventListener("click", renderFiltered);
    if (resetTraceBtn) {
        resetTraceBtn.addEventListener("click", () => {
            traceName.value = "";
            tracePhone.value = "";
            traceAddress.value = "";
            renderFiltered();
        });
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
