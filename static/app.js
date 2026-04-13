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

    // ── Global State ──
    let allResults = [];
    let auditCache = {}; // Stores { finding_id: result_object }

    // ── Batch processor  ──
    async function processBatch(findings) {
        // We do NOT clear allResults here anymore; we append/update
        // But for the grid display, we usually want to show the current "set"
        resultsGrid.innerHTML = "";
        resultsSection.classList.remove("hidden");
        progressSection.classList.remove("hidden");
        singleBtn.disabled = true;

        const total = findings.length;
        progressTitle.textContent = `Auditing ${total} records…`;

        // AUTO-APP DETECTION: Identify and set the app dropdown automatically
        if (findings.length > 0 && findings[0].app_name) {
            const detectedApp = findings[0].app_name.toLowerCase().replace(/ /g, "_");
            const appOption = Array.from(appSelect.options).find(opt => opt.value === detectedApp);
            if (appOption) {
                appSelect.value = detectedApp;
            }
        }

        // Local array just for this batch's display
        const batchResults = [];

        for (let i = 0; i < total; i++) {
            const finding = findings[i];
            const fid = finding.finding_id;

            progressCount.textContent = `${i + 1} / ${total}`;
            progressFill.style.width = `${((i + 1) / total) * 100}%`;

            // CACHE CHECK: If already audited, skip the LLM call
            if (auditCache[fid]) {
                const cachedRes = auditCache[fid];
                batchResults.push(cachedRes);
                
                // Ensure it's in allResults if not present
                if (!allResults.find(r => r.finding_id === fid)) {
                    allResults.push(cachedRes);
                }
                
                renderCard(cachedRes, i);
                continue;
            }

            try {
                const resp = await fetch("/api/verify-single", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(finding),
                });
                const result = await resp.json();
                
                // Save to cache
                auditCache[fid] = result;
                
                // Update allResults (avoid duplicates if re-auditing)
                const existingIdx = allResults.findIndex(r => r.finding_id === fid);
                if (existingIdx !== -1) allResults[existingIdx] = result;
                else allResults.push(result);

                batchResults.push(result);
                renderCard(result, i);
            } catch (err) {
                const errorResult = {
                    finding_id: fid || `F-${i}`,
                    app_name: finding.app_name || "Unknown",
                    status: "ERROR",
                    error: err.message,
                    answer: "",
                    evidence: [],
                };
                renderCard(errorResult, i);
            }
        }

        progressTitle.textContent = "Audit session complete!";
        singleBtn.disabled = false;
        updateStats();
    }

    // ── Render a result card (Clean Audit Dossier) ──
    function renderCard(result, index) {
        const card = document.createElement("div");
        card.className = "dossier-card";

        // Logic for mapping backend status to UI dossiers
        let statusClass = "status-insufficient";
        let statusLabel = "INSUFFICIENT INFORMATION";
        
        if (result.status === "VERIFIED" || result.status === "PARTIALLY_VERIFIED") {
            statusClass = "status-following";
            statusLabel = "FOLLOWING POLICY";
        } else if (result.status === "REJECTED" || result.status === "NOT_FOUND" || result.status === "ERROR") {
            statusClass = "status-not_following";
            statusLabel = "NOT FOLLOWING";
        }

        // Digital Footprint Table Rows
        const collectedEntries = Object.entries(result).filter(([k]) => k.startsWith("collected_"));
        const tableRows = collectedEntries.map(([k, v]) => {
            const label = k.replace("collected_", "").replace(/_/g, " ").toUpperCase();
            return `
                <tr>
                    <td class="attr-name">${escapeHtml(label)}</td>
                    <td class="attr-value">${escapeHtml(v)}</td>
                </tr>
            `;
        }).join("");

        // Footnotes / Evidence
        let footnotesHtml = "";
        if (result.evidence && result.evidence.length > 0) {
            const items = result.evidence.map((ev, i) =>
                `<div class="footnote-item">
                    <span class="footnote-ref">[${i + 1}]</span>
                    <div>
                        <span style="font-size: 10px; display: block; color: var(--accent-blue);">${escapeHtml(ev.chunk_id || ev.citation || "POLICY_REF")}</span>
                        ${escapeHtml(ev.text || ev.sentence || "")}
                    </div>
                </div>`
            ).join("");
            
            footnotesHtml = `
                <div class="footnotes">
                    <div class="footnotes-title">Policy Reference & Provenance</div>
                    ${items}
                </div>
            `;
        }

        card.innerHTML = `
            <div class="dossier-header">
                <span class="dossier-id">${escapeHtml(result.finding_id || "D-ID-UNK")}</span>
                <span class="dossier-status ${statusClass}">${statusLabel}</span>
            </div>
            <div class="dossier-body">
                <div class="incident-banner">
                    <span>Forensic Report: ${escapeHtml(result.app_name || "Unknown Interface")}</span>
                    <span>Incident Context: ${escapeHtml(result.collection_context || "Background Telemetry")}</span>
                </div>
                
                <div class="forensic-verdict">
                    <div class="verdict-title">⚐ Auditor's Verdict</div>
                    <div class="verdict-text">${formatAnswer(result.answer || "No narrative verdict provided by core engine.")}</div>
                </div>

                <div class="verdict-title">Captured Data Bundle (Digital Footprint)</div>
                <table class="dossier-table">
                    <thead>
                        <tr>
                            <th>Captured Attribute</th>
                            <th>Intercepted Value</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${tableRows || '<tr><td colspan="2">No data points captured in this bundle.</td></tr>'}
                    </tbody>
                </table>

                <div class="incident-banner" style="border: none; margin-top: 10px; opacity: 0.6;">
                    <span>Time of Recording: ${escapeHtml(result.timestamp || "N/A")}</span>
                </div>

                ${footnotesHtml}
            </div>
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
            <span class="stat-pill stat-pill-success">✓ ${following} Following Privacy Policy</span>
            <span class="stat-pill stat-pill-danger">✗ ${notFollowing} Not Following</span>
            <span class="stat-pill stat-pill-caution">⚠ ${insufficient} Insufficient Information</span>
        `;
    }

    // ── Identity Tracker ──
    const traceBtn = document.getElementById("traceBtn");
    const resetTraceBtn = document.getElementById("resetTraceBtn");
    const appSelect = document.getElementById("appSelect");
    const idTypeSelect = document.getElementById("idTypeSelect");
    const idValueInput = document.getElementById("idValueInput");
    const trackerSummary = document.getElementById("trackerSummary");

    async function handleTrace() {
        const appChoice = appSelect.value;
        const idType = idTypeSelect.value;
        const idValue = idValueInput.value.trim().toLowerCase();

        if (!appChoice) {
            alert("Please select an app to audit.");
            return;
        }
        if (!idValue) {
            alert("Please enter an identifier value to trace.");
            return;
        }

        // 1. Load the specific research data for this app if not already loaded or different
        // For simplicity, we'll reload the results set for the specific app
        try {
            const resp = await fetch(`/api/load-research-data/${appChoice}`);
            if (!resp.ok) throw new Error("Failed to load app data");
            const data = await resp.json();
            
            // 2. Process the batch (verify against policy)
            // Note: This triggers the LLM for the 30 rows
            await processBatch(data.findings);
            
            // 3. Apply Local Identity Filtering
            applyIdentityFilter(idType, idValue);
        } catch (err) {
            alert(`Error tracing app data: ${err.message}`);
        }
    }

    function applyIdentityFilter(targetKey, targetValue) {
        resultsGrid.innerHTML = "";
        trackerSummary.classList.add("hidden");

        // --- Recursive Forensic Linkage Algorithm ---
        
        let identifiedIndices = new Set();
        let knownValues = {
            collected_name: new Set(),
            collected_phone: new Set(),
            collected_email: new Set(),
            collected_pan: new Set(),
            collected_aadhaar: new Set(),
            collected_bank_account: new Set(),
            collected_login_id: new Set(),
            collected_device_id: new Set(),
            collected_advertisement_id: new Set(),
            collected_local_ip: new Set()
        };

        // 1. Initial Seed: Find records matching the user's search input
        allResults.forEach((res, idx) => {
            const val = (res[targetKey] || "").toLowerCase();
            if (val.includes(targetValue)) {
                identifiedIndices.add(idx);
                // Extract all known identity markers from this seed record
                for (let key in knownValues) {
                    if (res[key]) knownValues[key].add(res[key].toLowerCase());
                }
            }
        });

        if (identifiedIndices.size === 0) {
            trackerSummary.classList.remove("hidden");
            trackerSummary.innerHTML = `No records found for this identifier in the selected app.`;
            trackerSummary.style.background = "var(--glass-bg)";
            return;
        }

        // 2. Recursive Expansion: "Hop" from PII to Technical Bridges (Device ID, etc.)
        let addedNewMatch = true;
        while (addedNewMatch) {
            addedNewMatch = false;
            allResults.forEach((res, idx) => {
                if (identifiedIndices.has(idx)) return; // Already linked

                // Check if this row shares ANY known identity marker or biological/technical bridge
                let isMatch = false;
                for (let key in knownValues) {
                    const rowVal = (res[key] || "").toLowerCase();
                    if (rowVal && knownValues[key].has(rowVal)) {
                        isMatch = true;
                        break;
                    }
                }

                if (isMatch) {
                    identifiedIndices.add(idx);
                    addedNewMatch = true;
                    // Extract NEW markers from this newly linked record (Expand the graph)
                    for (let key in knownValues) {
                        if (res[key]) knownValues[key].add(res[key].toLowerCase());
                    }
                }
            });
        }

        // 3. Final Filter: Identify Privacy Violations (status !== VERIFIED)
        const finalResults = Array.from(identifiedIndices).map(idx => allResults[idx]);
        const violationResults = finalResults.filter(r => 
            r.status === "REJECTED" || r.status === "NOT_FOUND" || r.status === "ERROR"
        );

        violationResults.forEach((res, idx) => renderCard(res, idx));

        trackerSummary.classList.remove("hidden");
        if (violationResults.length > 0) {
            const appName = appSelect.options[appSelect.selectedIndex]?.text || "selected app";
            trackerSummary.innerHTML = `<strong>${violationResults.length} unauthorized findings</strong> detected within the ${appName} ecosystem.`;
            trackerSummary.style.background = "var(--red-bg)";
            trackerSummary.style.border = "1px solid rgba(239, 68, 68, 0.4)";
        } else {
            trackerSummary.innerHTML = `<strong>0 unauthorized privacy leaks</strong> detected.`;
            trackerSummary.style.background = "var(--glass-bg)";
            trackerSummary.style.border = "1px solid var(--border)";
        }
    }

    if (traceBtn) traceBtn.addEventListener("click", handleTrace);
    if (resetTraceBtn) {
        resetTraceBtn.addEventListener("click", () => {
            appSelect.selectedIndex = 0;
            idValueInput.value = "";
            resultsSection.classList.add("hidden");
            allResults = [];
            auditCache = {}; // Full reset
            resultsGrid.innerHTML = "";
            trackerSummary.classList.add("hidden");
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
        if (!text) return "";
        
        // Convert markdown-style sections and list to structured HTML
        return escapeHtml(text)
            .replace(/### (.*)/g, '<div style="margin-top: 20px; font-weight: 700; color: var(--text-primary); font-size: 11px; text-transform: uppercase;">$1</div>')
            .replace(/^\* /gm, "• ")
            .replace(/^- /gm, "• ")
            .replace(/• (.*)/g, '<div style="padding-left: 12px; margin-bottom: 4px;">• $1</div>')
            .replace(/\[([^\]]+)\]/g, '<strong style="color: var(--accent);">[$1]</strong>')
            .replace(/\n\n/g, '<div style="height: 12px;"></div>')
            .replace(/\n/g, "<br>");
    }
});
