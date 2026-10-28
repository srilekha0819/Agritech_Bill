let currentFile = null;
let currentRotation = 0;
let currentZoom = 1.0;
let lastStructuredJson = null;

// Setup Drag & Drop and File Picker
document.addEventListener("DOMContentLoaded", () => {
    const dropzone = document.getElementById("dropzone");
    const fileInput = document.getElementById("file-input");

    ['dragenter', 'dragover'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.remove('dragover');
        });
    });

    dropzone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files && files.length > 0) {
            handleFileSelect(files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });

});

function setRotation(deg) {
    currentRotation = deg;
    [0, 90, 180, 270].forEach(d => {
        const btn = document.getElementById(`rot-${d}`);
        if (btn) btn.classList.toggle('active', d === deg);
    });
    const previewImg = document.getElementById("bill-preview-img");
    if (previewImg) {
        previewImg.style.transform = `scale(${currentZoom}) rotate(${currentRotation}deg)`;
    }
}

function zoomImage(factor) {
    currentZoom *= factor;
    currentZoom = Math.max(0.5, Math.min(currentZoom, 3.0));
    const previewImg = document.getElementById("bill-preview-img");
    if (previewImg) {
        previewImg.style.transform = `scale(${currentZoom}) rotate(${currentRotation}deg)`;
    }
}

function resetZoom() {
    currentZoom = 1.0;
    const previewImg = document.getElementById("bill-preview-img");
    if (previewImg) {
        previewImg.style.transform = `scale(${currentZoom}) rotate(${currentRotation}deg)`;
    }
}

function handleFileSelect(file) {
    currentFile = file;

    // Highlight dropzone
    const dropzone = document.getElementById("dropzone");
    if (dropzone) {
        dropzone.classList.add("active-file");
        const sizeKb = (file.size / 1024).toFixed(1);
        dropzone.innerHTML = `
            <input type="file" id="file-input" accept="image/*" style="display:none;">
            <div class="dropzone-content" onclick="document.getElementById('file-input').click()">
                <div class="upload-icon">
                    <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path>
                        <polyline points="22 4 12 14.01 9 11.01"></polyline>
                    </svg>
                </div>
                <div class="file-selected-badge">✓ Selected: ${escapeHtml(file.name)} (${sizeKb} KB)</div>
                <p class="drop-sub" style="margin-top:8px;">Click or drop another document image to replace</p>
            </div>
        `;
        // Re-attach change listener
        const newFileInput = document.getElementById("file-input");
        if (newFileInput) {
            newFileInput.addEventListener('change', (e) => {
                if (e.target.files && e.target.files.length > 0) {
                    handleFileSelect(e.target.files[0]);
                }
            });
        }
    }

    // Update Process button label
    const procBtn = document.getElementById("btn-process");
    if (procBtn) {
        procBtn.innerHTML = `
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <polygon points="5 3 19 12 5 21 5 3"></polygon>
            </svg>
            Extract, Translate & Structure: ${escapeHtml(file.name.substring(0, 24))}
        `;
    }

    // Preview
    const reader = new FileReader();
    reader.onload = (e) => {
        const previewImg = document.getElementById("bill-preview-img");
        previewImg.src = e.target.result;
        resetZoom();
    };
    reader.readAsDataURL(file);
}

async function processCurrentImage() {
    const loadingOverlay = document.getElementById("loading-overlay");
    const kpiSection = document.getElementById("kpi-section");
    const resultsSection = document.getElementById("results-section");
    const loadingStatus = document.getElementById("loading-status");

    loadingOverlay.style.display = "block";

    try {
        let response;
        if (currentFile) {
            loadingStatus.textContent = `Extracting text & analyzing document: ${currentFile.name}...`;
            const formData = new FormData();
            formData.append("file", currentFile);
            formData.append("rotation", currentRotation);
            response = await fetch("/api/process", {
                method: "POST",
                body: formData
            });
        } else {
            alert("Please upload a document image first.");
            loadingOverlay.style.display = "none";
            return;
        }

        const data = await response.json();
        if (data.success) {
            renderResults(data);
            kpiSection.style.display = "grid";
            resultsSection.style.display = "grid";
        } else {
            alert("Error processing bill: " + (data.error || "Unknown error"));
        }
    } catch (err) {
        console.error("Pipeline error:", err);
        alert("Failed to process bill image: " + (err.message || err));
    } finally {
        loadingOverlay.style.display = "none";
    }
}

function renderResults(data) {
    const json = data.structured_json || {};
    lastStructuredJson = json;

    // 1. KPIs
    const meta = json.document_metadata || {};
    const parties = json.stakeholders || {};
    const items = (json.line_items && json.line_items.length > 0) ? json.line_items : [{}];
    const settlement = json.financial_settlement || {};

    const committeeEl = document.getElementById("kpi-committee");
    if (committeeEl) committeeEl.textContent = meta.market_committee || "-";
    const billDateEl = document.getElementById("kpi-bill-date");
    if (billDateEl) billDateEl.textContent = `No: ${meta.bill_number || '-'} • ${meta.date || '-'}`;
    const buyerEl = document.getElementById("kpi-buyer");
    if (buyerEl) buyerEl.textContent = parties.buyer || "-";
    const farmerEl = document.getElementById("kpi-farmer");
    if (farmerEl) farmerEl.textContent = `${parties.farmer_seller || '-'} (${parties.village || '-'})`;
    
    const bagText = items[0] && items[0].bag_count ? `${items[0].bag_count} bags` : "";
    const wtText = items[0] && items[0].net_weight_quintals ? `${items[0].net_weight_quintals} Qtl` : "";
    const weightEl = document.getElementById("kpi-weight");
    if (weightEl) weightEl.textContent = `${wtText} ${bagText ? '• ' + bagText : ''}`.trim() || "-";

    const payableEl = document.getElementById("kpi-net-payable");
    if (payableEl) {
        const payable = settlement.net_payable_to_farmer && !isNaN(Number(settlement.net_payable_to_farmer))
            ? `₹ ${Number(settlement.net_payable_to_farmer).toLocaleString('en-IN')}` 
            : (settlement.net_payable_to_farmer || "-");
        payableEl.textContent = payable;
    }
    const confidence = data.ocr_confidence_estimate;
    document.getElementById("kpi-ocr-confidence").textContent =
        Number.isFinite(confidence) ? `${(confidence * 100).toFixed(1)}%` : "Not available";

    // 2. Extracted Multilingual Lines
    const ocrContainer = document.getElementById("ocr-results-container");
    const lines = data.lines || [];
    document.getElementById("ocr-line-count").textContent = `${lines.length} Lines Detected`;

    if (lines.length === 0) {
        ocrContainer.innerHTML = '<div class="placeholder-text">No text lines detected.</div>';
    } else {
        ocrContainer.innerHTML = lines.map(line => `
            <div class="line-item">
                <span class="line-num">#${line.line_id}</span>
                <span class="line-text">${escapeHtml(line.original_text || line.text || '')}</span>
            </div>
        `).join("");
    }

    // 3. Translated English Lines
    const transContainer = document.getElementById("translated-results-container");
    if (lines.length === 0) {
        transContainer.innerHTML = '<div class="placeholder-text">No translation available.</div>';
    } else {
        transContainer.innerHTML = lines.map(line => `
            <div class="line-item">
                <span class="line-num">#${line.line_id}</span>
                <span class="line-text">${escapeHtml(line.translated_text || '')}</span>
            </div>
        `).join("");
    }

    // 4. Structured JSON Viewer
    const jsonOutput = document.getElementById("json-output");
    jsonOutput.innerHTML = syntaxHighlight(JSON.stringify(json, null, 2));
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function syntaxHighlight(json) {
    json = json.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    return json.replace(/("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)/g, function (match) {
        let cls = 'number';
        if (/^"/.test(match)) {
            if (/:$/.test(match)) {
                cls = 'key';
                return `<span style="color:#a78bfa">${match}</span>`;
            } else {
                cls = 'string';
                return `<span style="color:#38bdf8">${match}</span>`;
            }
        } else if (/true|false/.test(match)) {
            return `<span style="color:#f472b6">${match}</span>`;
        } else if (/null/.test(match)) {
            return `<span style="color:#94a3b8">${match}</span>`;
        }
        return `<span style="color:#34d399">${match}</span>`;
    });
}

function copyJSON() {
    if (!lastStructuredJson) return;
    navigator.clipboard.writeText(JSON.stringify(lastStructuredJson, null, 2))
        .then(() => alert("Structured JSON copied to clipboard!"))
        .catch(() => alert("Failed to copy JSON."));
}

function downloadJSON() {
    if (!lastStructuredJson) return;
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(lastStructuredJson, null, 2));
    const downloadAnchor = document.createElement('a');
    downloadAnchor.setAttribute("href", dataStr);
    downloadAnchor.setAttribute("download", `bill_structured_${Date.now()}.json`);
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
}

function copyText(containerId) {
    const text = document.getElementById(containerId).innerText;
    navigator.clipboard.writeText(text)
        .then(() => alert("Copied text to clipboard!"))
        .catch(() => alert("Failed to copy text."));
}
