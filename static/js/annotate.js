/**
 * Precision Annotation Studio Client
 * High-accuracy manual labeling, live inspection, interactive canvas with 8-handle transform,
 * pan & zoom, rotation preservation, and Telugu virtual keyboard helper.
 */

(function () {
    "use strict";

    // State Variables
    let selectedImageFile = null;
    let existingFilename = null;
    let sourceImage = null;
    let rotation = 0; // 0, 90, 180, 270
    let contrastBoost = false;
    let zoom = 1.0;
    let panOffset = { x: 0, y: 0 };
    let toolMode = "draw"; // "draw" or "pan"
    let imageAnnotations = [];
    let selectedCropIndex = -1;

    // Interaction State
    let activeDrag = null; // { mode: "new"|"move"|"resize"|"pan", handle: string, startX, startY, origBbox, origPan }
    let isSpacePressed = false;
    let availableBills = [];

    // DOM Elements
    const canvas = document.getElementById("annotation-canvas");
    const ctx = canvas.getContext("2d");
    const viewport = document.getElementById("viewport-container");
    const transformer = document.getElementById("canvas-transformer");
    const emptyState = document.getElementById("canvas-empty-state");

    const imageInput = document.getElementById("image-input");
    const emptyImageInput = document.getElementById("empty-image-input");
    const btnGalleryPick = document.getElementById("btn-gallery-pick");
    const btnEmptyGallery = document.getElementById("btn-empty-gallery");
    const existingCountSpan = document.getElementById("existing-count");

    // Rotation & Tool Buttons
    const btnRotateCw = document.getElementById("btn-rotate-cw");
    const btnRotateCcw = document.getElementById("btn-rotate-ccw");
    const btnRotate180 = document.getElementById("btn-rotate-180");
    const rotationBadge = document.getElementById("rotation-badge");

    const btnToolDraw = document.getElementById("btn-tool-draw");
    const btnToolPan = document.getElementById("btn-tool-pan");
    const btnZoomIn = document.getElementById("btn-zoom-in");
    const btnZoomOut = document.getElementById("btn-zoom-out");
    const btnZoomFit = document.getElementById("btn-zoom-fit");
    const btnZoomReset = document.getElementById("btn-zoom-reset");
    const zoomReadout = document.getElementById("zoom-readout");

    const btnTightenInk = document.getElementById("btn-tighten-ink");
    const btnContrastBoost = document.getElementById("btn-contrast-boost");
    const applyLayoutButton = document.getElementById("apply-layout");
    const btnDeleteCrop = document.getElementById("btn-delete-crop");

    // HUD & Meta
    const currentFilenameDisplay = document.getElementById("current-filename-display");
    const hudCoords = document.getElementById("hud-coords");
    const imageDimensionsSpan = document.getElementById("image-dimensions");
    const cropsSummaryBadge = document.getElementById("crops-summary-badge");
    const cropCounterSpan = document.getElementById("crop-counter");

    // Inspector
    const activeCropBadge = document.getElementById("active-crop-badge");
    const activeCropTitle = document.getElementById("active-crop-title");
    const activeCropDimensions = document.getElementById("active-crop-dimensions");
    const cropPreviewCanvas = document.getElementById("crop-preview-canvas");
    const previewCtx = cropPreviewCanvas.getContext("2d");
    const previewPlaceholder = document.getElementById("preview-placeholder");

    const cropTypeControl = document.getElementById("crop-type-control");
    const fieldNameSelect = document.getElementById("field-name-select");
    const fieldNameCustom = document.getElementById("field-name-custom");
    const transcriptionInput = document.getElementById("transcription-input");
    const translationInput = document.getElementById("translation-input");
    const languageSelect = document.getElementById("language-select");

    const annotationCardsContainer = document.getElementById("annotation-cards-container");
    const billGroupIdInput = document.getElementById("bill-group-id");
    const saveButton = document.getElementById("save-annotations");
    const saveStatus = document.getElementById("save-status");

    // Modal
    const galleryModal = document.getElementById("gallery-modal");
    const modalCloseBtn = document.getElementById("modal-close-btn");
    const galleryGrid = document.getElementById("gallery-grid");
    const gallerySearch = document.getElementById("gallery-search");

    // Telugu Virtual Keyboard
    const toggleKbBtn = document.getElementById("toggle-kb-btn");
    const teluguKeysContainer = document.getElementById("telugu-keys-container");

    // ==========================================
    // INITIALIZATION & BILLS LIST
    // ==========================================
    async function init() {
        fetchUploadsList();
        setupEventListeners();
        setupTeluguKeyboard();
        setupFieldChips();
    }

    async function fetchUploadsList() {
        try {
            const res = await fetch("/api/uploads/list");
            const data = await res.json();
            if (data.success && Array.isArray(data.images)) {
                availableBills = data.images;
                if (existingCountSpan) existingCountSpan.textContent = String(availableBills.length);
            }
        } catch (err) {
            console.warn("Could not fetch uploads list:", err);
        }
    }

    // ==========================================
    // IMAGE LOADING & ROTATION
    // ==========================================
    function loadImageFromFile(file) {
        if (!file) return;
        selectedImageFile = file;
        existingFilename = null;
        imageAnnotations = [];
        selectedCropIndex = -1;
        rotation = 0;
        contrastBoost = false;

        const baseStem = file.name.replace(/\.[^.]+$/, "");
        billGroupIdInput.value = baseStem;
        currentFilenameDisplay.textContent = file.name;

        const objectUrl = URL.createObjectURL(file);
        const img = new Image();
        img.onload = () => {
            sourceImage = img;
            URL.revokeObjectURL(objectUrl);
            onImageLoaded();
        };
        img.onerror = () => {
            URL.revokeObjectURL(objectUrl);
            setSaveStatus("Failed to load uploaded image file.", "error");
        };
        img.src = objectUrl;
    }

    function loadImageFromExisting(filename) {
        if (!filename) return;
        selectedImageFile = null;
        existingFilename = filename;
        imageAnnotations = [];
        selectedCropIndex = -1;
        rotation = 0;
        contrastBoost = false;

        const baseStem = filename.replace(/\.[^.]+$/, "");
        billGroupIdInput.value = baseStem;
        currentFilenameDisplay.textContent = filename;

        const img = new Image();
        img.crossOrigin = "anonymous";
        img.onload = () => {
            sourceImage = img;
            onImageLoaded();
            closeGalleryModal();
        };
        img.onerror = () => {
            setSaveStatus("Failed to load bill image from server.", "error");
        };
        img.src = `/api/uploads/raw/${encodeURIComponent(filename)}`;
    }

    function onImageLoaded() {
        emptyState.style.display = "none";
        rotationBadge.textContent = `${rotation}°`;
        updateCanvasDimensions();
        zoomToFit();
        drawCanvas();
        renderAnnotationCards();
        updateInspector();
        applyLayoutButton.disabled = false;
        saveButton.disabled = imageAnnotations.length === 0;
    }

    function updateCanvasDimensions() {
        if (!sourceImage) return;
        if (rotation === 90 || rotation === 270) {
            canvas.width = sourceImage.naturalHeight;
            canvas.height = sourceImage.naturalWidth;
        } else {
            canvas.width = sourceImage.naturalWidth;
            canvas.height = sourceImage.naturalHeight;
        }
        imageDimensionsSpan.textContent = `${canvas.width} × ${canvas.height} px`;
    }

    function rotateImage(deltaAngle) {
        if (!sourceImage) return;
        const oldW = canvas.width;
        const oldH = canvas.height;

        rotation = (rotation + deltaAngle + 360) % 360;
        rotationBadge.textContent = `${rotation}°`;
        updateCanvasDimensions();

        // Transform existing bounding boxes so they stay pinned to visual features!
        const newW = canvas.width;
        const newH = canvas.height;
        imageAnnotations.forEach((ann) => {
            const [x, y, w, h] = ann.bbox;
            let nx, ny, nw, nh;
            if (deltaAngle === 90 || deltaAngle === -270) {
                // Clockwise 90: (x, y) -> (oldH - (y + h), x), (w, h) -> (h, w)
                nx = oldH - (y + h);
                ny = x;
                nw = h;
                nh = w;
            } else if (deltaAngle === -90 || deltaAngle === 270) {
                // Counter-clockwise 90: (x, y) -> (y, oldW - (x + w)), (w, h) -> (h, w)
                nx = y;
                ny = oldW - (x + w);
                nw = h;
                nh = w;
            } else if (deltaAngle === 180 || deltaAngle === -180) {
                nx = oldW - (x + w);
                ny = oldH - (y + h);
                nw = w;
                nh = h;
            }
            ann.bbox = [Math.round(nx), Math.round(ny), Math.round(nw), Math.round(nh)];
        });

        drawCanvas();
        updateInspector();
        renderAnnotationCards();
    }

    // ==========================================
    // VIEWPORT TRANSFORMATION & ZOOM
    // ==========================================
    function applyTransform() {
        transformer.style.transform = `translate(${panOffset.x}px, ${panOffset.y}px) scale(${zoom})`;
        zoomReadout.textContent = `${Math.round(zoom * 100)}%`;
    }

    function zoomToFit() {
        if (!canvas.width || !canvas.height) return;
        const viewW = viewport.clientWidth;
        const viewH = viewport.clientHeight;
        const scaleX = (viewW - 40) / canvas.width;
        const scaleY = (viewH - 40) / canvas.height;
        zoom = Math.max(0.15, Math.min(2.5, Math.min(scaleX, scaleY)));
        panOffset.x = Math.round((viewW - canvas.width * zoom) / 2);
        panOffset.y = Math.round((viewH - canvas.height * zoom) / 2);
        applyTransform();
    }

    function zoomBy(factor, centerPoint = null) {
        if (!sourceImage) return;
        const oldZoom = zoom;
        const nextZoom = Math.max(0.15, Math.min(6.0, zoom * factor));
        if (nextZoom === oldZoom) return;

        const viewRect = viewport.getBoundingClientRect();
        const pivotX = centerPoint ? centerPoint.x : viewRect.width / 2;
        const pivotY = centerPoint ? centerPoint.y : viewRect.height / 2;

        panOffset.x = pivotX - (pivotX - panOffset.x) * (nextZoom / oldZoom);
        panOffset.y = pivotY - (pivotY - panOffset.y) * (nextZoom / oldZoom);
        zoom = nextZoom;
        applyTransform();
    }

    function screenToCanvas(screenX, screenY) {
        const viewRect = viewport.getBoundingClientRect();
        const clientRelX = screenX - viewRect.left;
        const clientRelY = screenY - viewRect.top;
        const canvasX = (clientRelX - panOffset.x) / zoom;
        const canvasY = (clientRelY - panOffset.y) / zoom;
        return {
            x: Math.max(0, Math.min(canvas.width, canvasX)),
            y: Math.max(0, Math.min(canvas.height, canvasY)),
        };
    }

    // ==========================================
    // CANVAS RENDERING
    // ==========================================
    const HANDLE_SIZE = 9;

    function getBoxHandles(x, y, w, h) {
        return {
            nw: { x: x, y: y },
            n: { x: x + w / 2, y: y },
            ne: { x: x + w, y: y },
            e: { x: x + w, y: y + h / 2 },
            se: { x: x + w, y: y + h },
            s: { x: x + w / 2, y: y + h },
            sw: { x: x, y: y + h },
            w: { x: x, y: y + h / 2 },
        };
    }

    function drawCanvas() {
        if (!sourceImage) return;
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Draw rotated source image
        ctx.save();
        if (contrastBoost) {
            ctx.filter = "contrast(180%) brightness(105%)";
        }
        if (rotation === 90) {
            ctx.translate(canvas.width, 0);
            ctx.rotate(Math.PI / 2);
        } else if (rotation === 180) {
            ctx.translate(canvas.width, canvas.height);
            ctx.rotate(Math.PI);
        } else if (rotation === 270) {
            ctx.translate(0, canvas.height);
            ctx.rotate(-Math.PI / 2);
        }
        ctx.drawImage(sourceImage, 0, 0);
        ctx.restore();

        // Draw all bounding boxes
        imageAnnotations.forEach((ann, index) => {
            const isSelected = index === selectedCropIndex;
            const [x, y, w, h] = ann.bbox;

            let strokeColor = "#06b6d4"; // cyan for printed
            let fillColor = "rgba(6, 182, 212, 0.15)";
            if (ann.type === "handwritten") {
                strokeColor = "#10b981"; // emerald
                fillColor = "rgba(16, 185, 129, 0.18)";
            } else if (ann.type === "digits") {
                strokeColor = "#f59e0b"; // amber
                fillColor = "rgba(245, 158, 11, 0.18)";
            }

            if (isSelected) {
                strokeColor = "#fbbf24";
                fillColor = "rgba(251, 191, 36, 0.22)";
            }

            // Fill & border
            ctx.fillStyle = fillColor;
            ctx.fillRect(x, y, w, h);

            ctx.lineWidth = isSelected ? Math.max(2.5, 2.5 / zoom) : Math.max(1.8, 1.8 / zoom);
            ctx.strokeStyle = strokeColor;
            ctx.strokeRect(x, y, w, h);

            // Floating Header Badge above crop
            const labelText = `${index + 1}. ${ann.field_name || "Unassigned"}`;
            const fontSize = Math.max(13, Math.min(18, Math.round(14 / Math.min(zoom, 1.5))));
            ctx.font = `600 ${fontSize}px var(--font-body)`;
            const textWidth = ctx.measureText(labelText).width;
            const badgeH = fontSize + 8;
            const badgeW = textWidth + 12;
            const badgeY = Math.max(0, y - badgeH - 2);

            ctx.fillStyle = isSelected ? "#f59e0b" : "#0f172a";
            ctx.fillRect(x, badgeY, badgeW, badgeH);
            ctx.fillStyle = isSelected ? "#090d16" : "#ffffff";
            ctx.fillText(labelText, x + 6, badgeY + fontSize);

            // If selected, draw 8 resize handles
            if (isSelected) {
                const handles = getBoxHandles(x, y, w, h);
                const handleR = Math.max(HANDLE_SIZE / 2, (HANDLE_SIZE / 2) / zoom);
                ctx.fillStyle = "#fbbf24";
                ctx.strokeStyle = "#090d16";
                ctx.lineWidth = Math.max(1.5, 1.5 / zoom);

                Object.values(handles).forEach((pt) => {
                    ctx.beginPath();
                    ctx.rect(pt.x - handleR, pt.y - handleR, handleR * 2, handleR * 2);
                    ctx.fill();
                    ctx.stroke();
                });
            }
        });
    }

    // ==========================================
    // HIT TESTING & MOUSE INTERACTION
    // ==========================================
    function hitTestHandle(pt, bbox) {
        const [x, y, w, h] = bbox;
        const handles = getBoxHandles(x, y, w, h);
        const threshold = Math.max(9, 9 / zoom);
        for (const [name, hpt] of Object.entries(handles)) {
            if (Math.abs(pt.x - hpt.x) <= threshold && Math.abs(pt.y - hpt.y) <= threshold) {
                return name;
            }
        }
        return null;
    }

    function hitTestBox(pt) {
        // Test in reverse order (topmost first)
        for (let i = imageAnnotations.length - 1; i >= 0; i--) {
            const [x, y, w, h] = imageAnnotations[i].bbox;
            if (pt.x >= x && pt.x <= x + w && pt.y >= y && pt.y <= y + h) {
                return i;
            }
        }
        return -1;
    }

    viewport.addEventListener("pointerdown", (e) => {
        if (!sourceImage) return;
        viewport.setPointerCapture(e.pointerId);

        // Check if middle click, spacebar, or pan tool mode
        if (e.button === 1 || isSpacePressed || toolMode === "pan") {
            activeDrag = {
                mode: "pan",
                startX: e.clientX,
                startY: e.clientY,
                origPan: { ...panOffset },
            };
            viewport.classList.add("mode-pan");
            return;
        }

        if (e.button !== 0) return; // Left click only

        const pt = screenToCanvas(e.clientX, e.clientY);

        // 1. Check if clicked a handle of the currently selected box
        if (selectedCropIndex >= 0 && selectedCropIndex < imageAnnotations.length) {
            const selectedBox = imageAnnotations[selectedCropIndex].bbox;
            const handle = hitTestHandle(pt, selectedBox);
            if (handle) {
                activeDrag = {
                    mode: "resize",
                    handle: handle,
                    startPt: pt,
                    origBbox: [...selectedBox],
                };
                return;
            }
        }

        // 2. Check if clicked inside an existing box
        const hitIndex = hitTestBox(pt);
        if (hitIndex >= 0) {
            selectCrop(hitIndex);
            activeDrag = {
                mode: "move",
                startPt: pt,
                origBbox: [...imageAnnotations[hitIndex].bbox],
            };
            drawCanvas();
            return;
        }

        // 3. Clicked empty space -> start drawing new box
        activeDrag = {
            mode: "new",
            startPt: pt,
            origBbox: [Math.round(pt.x), Math.round(pt.y), 1, 1],
        };
        // Deselect current
        selectCrop(-1);
    });

    viewport.addEventListener("pointermove", (e) => {
        if (!sourceImage) return;
        const pt = screenToCanvas(e.clientX, e.clientY);
        hudCoords.textContent = `X: ${Math.round(pt.x)} Y: ${Math.round(pt.y)}`;

        if (!activeDrag) {
            updateCursor(pt);
            return;
        }

        if (activeDrag.mode === "pan") {
            const dx = e.clientX - activeDrag.startX;
            const dy = e.clientY - activeDrag.startY;
            panOffset.x = activeDrag.origPan.x + dx;
            panOffset.y = activeDrag.origPan.y + dy;
            applyTransform();
            return;
        }

        if (activeDrag.mode === "new") {
            const x0 = Math.min(activeDrag.startPt.x, pt.x);
            const y0 = Math.min(activeDrag.startPt.y, pt.y);
            const w = Math.abs(pt.x - activeDrag.startPt.x);
            const h = Math.abs(pt.y - activeDrag.startPt.y);

            drawCanvas();
            // Draw temporary selection box
            ctx.save();
            ctx.strokeStyle = "#f59e0b";
            ctx.lineWidth = Math.max(2, 2 / zoom);
            ctx.setLineDash([8, 4]);
            ctx.strokeRect(x0, y0, w, h);
            ctx.fillStyle = "rgba(245, 158, 11, 0.2)";
            ctx.fillRect(x0, y0, w, h);
            ctx.restore();
            hudCoords.textContent = `X: ${Math.round(x0)} Y: ${Math.round(y0)} W: ${Math.round(w)} H: ${Math.round(h)}`;
            return;
        }

        if (activeDrag.mode === "move" && selectedCropIndex >= 0) {
            const dx = pt.x - activeDrag.startPt.x;
            const dy = pt.y - activeDrag.startPt.y;
            const [ox, oy, ow, oh] = activeDrag.origBbox;
            const nx = Math.max(0, Math.min(canvas.width - ow, ox + dx));
            const ny = Math.max(0, Math.min(canvas.height - oh, oy + dy));
            imageAnnotations[selectedCropIndex].bbox = [Math.round(nx), Math.round(ny), ow, oh];
            drawCanvas();
            updateInspector();
            return;
        }

        if (activeDrag.mode === "resize" && selectedCropIndex >= 0) {
            const [ox, oy, ow, oh] = activeDrag.origBbox;
            const dx = pt.x - activeDrag.startPt.x;
            const dy = pt.y - activeDrag.startPt.y;
            let nx = ox, ny = oy, nw = ow, nh = oh;

            const hName = activeDrag.handle;
            if (hName.includes("e")) nw = Math.max(8, ow + dx);
            if (hName.includes("s")) nh = Math.max(8, oh + dy);
            if (hName.includes("w")) {
                const candW = ow - dx;
                if (candW >= 8) {
                    nx = ox + dx;
                    nw = candW;
                }
            }
            if (hName.includes("n")) {
                const candH = oh - dy;
                if (candH >= 8) {
                    ny = oy + dy;
                    nh = candH;
                }
            }

            imageAnnotations[selectedCropIndex].bbox = [
                Math.round(Math.max(0, Math.min(canvas.width - nw, nx))),
                Math.round(Math.max(0, Math.min(canvas.height - nh, ny))),
                Math.round(nw),
                Math.round(nh),
            ];
            drawCanvas();
            updateInspector();
            return;
        }
    });

    viewport.addEventListener("pointerup", (e) => {
        if (!activeDrag) return;
        const pt = screenToCanvas(e.clientX, e.clientY);

        if (activeDrag.mode === "new") {
            const x0 = Math.floor(Math.min(activeDrag.startPt.x, pt.x));
            const y0 = Math.floor(Math.min(activeDrag.startPt.y, pt.y));
            const w = Math.ceil(Math.abs(pt.x - activeDrag.startPt.x));
            const h = Math.ceil(Math.abs(pt.y - activeDrag.startPt.y));

            if (w >= 10 && h >= 10) {
                // Add new annotation
                imageAnnotations.push({
                    field_name: "",
                    field_category: "custom",
                    text: "",
                    translated_text: "",
                    type: "printed",
                    language: "te",
                    bbox: [x0, y0, w, h],
                });
                selectCrop(imageAnnotations.length - 1);
                renderAnnotationCards();
                saveButton.disabled = false;
            }
        }

        activeDrag = null;
        viewport.classList.remove("mode-pan");
        drawCanvas();
        updateInspector();
    });

    // Zoom on wheel centered at mouse
    viewport.addEventListener("wheel", (e) => {
        e.preventDefault();
        const factor = e.deltaY < 0 ? 1.15 : 0.87;
        const viewRect = viewport.getBoundingClientRect();
        zoomBy(factor, { x: e.clientX - viewRect.left, y: e.clientY - viewRect.top });
    }, { passive: false });

    function updateCursor(pt) {
        if (toolMode === "pan" || isSpacePressed) {
            viewport.style.cursor = "grab";
            return;
        }
        if (selectedCropIndex >= 0 && selectedCropIndex < imageAnnotations.length) {
            const hName = hitTestHandle(pt, imageAnnotations[selectedCropIndex].bbox);
            if (hName) {
                const map = {
                    nw: "nwse-resize",
                    se: "nwse-resize",
                    ne: "nesw-resize",
                    sw: "nesw-resize",
                    n: "ns-resize",
                    s: "ns-resize",
                    e: "ew-resize",
                    w: "ew-resize",
                };
                viewport.style.cursor = map[hName] || "pointer";
                return;
            }
        }
        const hitIdx = hitTestBox(pt);
        if (hitIdx >= 0) {
            viewport.style.cursor = hitIdx === selectedCropIndex ? "move" : "pointer";
            return;
        }
        viewport.style.cursor = "crosshair";
    }

    // ==========================================
    // INSPECTOR & LIVE CROP PREVIEW
    // ==========================================
    function selectCrop(index) {
        selectedCropIndex = index;
        btnTightenInk.disabled = index < 0;
        btnDeleteCrop.disabled = index < 0;
        updateInspector();
        drawCanvas();
        highlightActiveCard();
    }

    function updateInspector() {
        if (selectedCropIndex < 0 || selectedCropIndex >= imageAnnotations.length) {
            activeCropBadge.textContent = "Select a Crop";
            activeCropBadge.className = "panel-tag tag-cyan";
            activeCropTitle.textContent = "Box Inspector";
            activeCropDimensions.textContent = "0 × 0 px";
            previewPlaceholder.style.display = "block";
            cropPreviewCanvas.style.display = "none";
            fieldNameSelect.value = "";
            fieldNameCustom.style.display = "none";
            fieldNameCustom.value = "";
            transcriptionInput.value = "";
            translationInput.value = "";
            languageSelect.value = "te";
            setSegmentedType("printed");
            return;
        }

        const ann = imageAnnotations[selectedCropIndex];
        const [x, y, w, h] = ann.bbox;

        activeCropBadge.textContent = `Crop #${selectedCropIndex + 1}`;
        activeCropTitle.textContent = ann.field_name || "Unassigned Field";
        activeCropDimensions.textContent = `${w} × ${h} px`;

        // Update inputs
        let matchedOption = false;
        for (const opt of fieldNameSelect.options) {
            if (opt.value === ann.field_name) {
                fieldNameSelect.value = ann.field_name;
                matchedOption = true;
                break;
            }
        }
        if (!matchedOption && ann.field_name) {
            fieldNameSelect.value = "CUSTOM";
            fieldNameCustom.style.display = "block";
            fieldNameCustom.value = ann.field_name;
        } else if (matchedOption) {
            fieldNameCustom.style.display = "none";
        }

        transcriptionInput.value = ann.text || "";
        translationInput.value = ann.translated_text || "";
        languageSelect.value = ann.language || "te";
        setSegmentedType(ann.type || "printed");

        // Render Live Magnified Crop Preview
        renderCropPreview(x, y, w, h);
    }

    function renderCropPreview(x, y, w, h) {
        if (w <= 0 || h <= 0) return;
        previewPlaceholder.style.display = "none";
        cropPreviewCanvas.style.display = "block";

        cropPreviewCanvas.width = Math.min(640, Math.max(160, w * 2));
        cropPreviewCanvas.height = Math.min(180, Math.max(48, h * 2));
        previewCtx.clearRect(0, 0, cropPreviewCanvas.width, cropPreviewCanvas.height);
        previewCtx.imageSmoothingQuality = "high";
        previewCtx.drawImage(
            canvas,
            x, y, w, h,
            0, 0, cropPreviewCanvas.width, cropPreviewCanvas.height
        );
    }

    function setSegmentedType(type) {
        cropTypeControl.querySelectorAll(".segment-btn").forEach((btn) => {
            btn.classList.toggle("active", btn.dataset.type === type);
        });
    }

    // ==========================================
    // TIGHTEN TO INK ALGORITHM
    // ==========================================
    function tightenActiveBoxToInk() {
        if (selectedCropIndex < 0 || selectedCropIndex >= imageAnnotations.length) return;
        const [x, y, w, h] = imageAnnotations[selectedCropIndex].bbox;
        if (w < 8 || h < 8) return;

        // Read pixel data from the canvas
        const imgData = ctx.getImageData(x, y, w, h);
        const data = imgData.data;

        // Compute average luminance to find foreground vs background
        let minX = w, maxX = 0, minY = h, maxY = 0;
        let count = 0;

        // Find ink pixels (significantly darker than white background)
        for (let py = 0; py < h; py++) {
            for (let px = 0; px < w; px++) {
                const idx = (py * w + px) * 4;
                const r = data[idx];
                const g = data[idx + 1];
                const b = data[idx + 2];
                const lum = 0.299 * r + 0.587 * g + 0.114 * b;
                if (lum < 165) { // Dark ink pixel threshold
                    if (px < minX) minX = px;
                    if (px > maxX) maxX = px;
                    if (py < minY) minY = py;
                    if (py > maxY) maxY = py;
                    count++;
                }
            }
        }

        if (count > 20 && maxX > minX && maxY > minY) {
            const pad = 3;
            const newX = Math.max(0, x + minX - pad);
            const newY = Math.max(0, y + minY - pad);
            const newW = Math.min(canvas.width - newX, (maxX - minX) + 2 * pad);
            const newH = Math.min(canvas.height - newY, (maxY - minY) + 2 * pad);

            imageAnnotations[selectedCropIndex].bbox = [
                Math.round(newX),
                Math.round(newY),
                Math.round(newW),
                Math.round(newH),
            ];
            drawCanvas();
            updateInspector();
            renderAnnotationCards();
            setSaveStatus("Snapped crop tightly to text ink.", "success");
        } else {
            setSaveStatus("Could not isolate ink contours inside this box.", "normal");
        }
    }

    // ==========================================
    // ALL CROPS LIST CARDS
    // ==========================================
    function renderAnnotationCards() {
        annotationCardsContainer.replaceChildren();
        const count = imageAnnotations.length;
        cropsSummaryBadge.textContent = `${count} crops`;
        cropCounterSpan.textContent = `${count} crops`;
        saveButton.disabled = count === 0;

        if (count === 0) {
            const empty = document.createElement("div");
            empty.className = "empty-list-hint";
            empty.textContent = "Draw boxes on the bill to add annotations.";
            annotationCardsContainer.appendChild(empty);
            return;
        }

        imageAnnotations.forEach((ann, index) => {
            const card = document.createElement("div");
            card.className = `crop-card ${index === selectedCropIndex ? "active" : ""}`;
            card.dataset.index = index;

            // Thumbnail
            const thumbWrap = document.createElement("div");
            thumbWrap.className = "crop-card-thumb";
            const thumbImg = document.createElement("canvas");
            thumbImg.width = 56;
            thumbImg.height = 40;
            const tCtx = thumbImg.getContext("2d");
            const [x, y, w, h] = ann.bbox;
            if (canvas.width > 0 && w > 0 && h > 0) {
                tCtx.drawImage(canvas, x, y, w, h, 0, 0, 56, 40);
            }
            thumbWrap.appendChild(thumbImg);

            // Info
            const info = document.createElement("div");
            info.className = "crop-card-info";

            const top = document.createElement("div");
            top.className = "crop-card-top";

            const num = document.createElement("span");
            num.className = "crop-card-num";
            num.textContent = `#${index + 1}`;

            const chip = document.createElement("span");
            chip.className = `crop-type-chip chip-${ann.type}`;
            chip.textContent = ann.type === "handwritten" ? "✍️ Hand" : ann.type === "digits" ? "🔢 Digits" : "🏷️ Label";

            top.append(num, chip);

            const name = document.createElement("div");
            name.className = "crop-card-name";
            name.textContent = ann.field_name || "Unassigned Field";

            const text = document.createElement("div");
            text.className = "crop-card-text";
            text.textContent = ann.text ? `"${ann.text}"` : "(No transcription yet)";

            info.append(top, name, text);

            // Delete button
            const delBtn = document.createElement("button");
            delBtn.className = "crop-card-delete";
            delBtn.type = "button";
            delBtn.innerHTML = "×";
            delBtn.title = `Delete crop #${index + 1}`;
            delBtn.addEventListener("click", (e) => {
                e.stopPropagation();
                deleteCrop(index);
            });

            card.append(thumbWrap, info, delBtn);
            card.addEventListener("click", () => selectCrop(index));
            annotationCardsContainer.appendChild(card);
        });
    }

    function highlightActiveCard() {
        annotationCardsContainer.querySelectorAll(".crop-card").forEach((card, i) => {
            card.classList.toggle("active", i === selectedCropIndex);
        });
    }

    function deleteCrop(index) {
        if (index < 0 || index >= imageAnnotations.length) return;
        imageAnnotations.splice(index, 1);
        if (selectedCropIndex >= imageAnnotations.length) {
            selectedCropIndex = imageAnnotations.length - 1;
        }
        selectCrop(selectedCropIndex);
        renderAnnotationCards();
        drawCanvas();
    }

    // ==========================================
    // TELUGU VIRTUAL KEYBOARD & QUICK CHIPS
    // ==========================================
    function setupTeluguKeyboard() {
        toggleKbBtn.addEventListener("click", () => {
            const isHidden = teluguKeysContainer.style.display === "none";
            teluguKeysContainer.style.display = isHidden ? "flex" : "none";
            toggleKbBtn.textContent = isHidden ? "Hide Characters" : "Show Characters";
        });

        teluguKeysContainer.querySelectorAll(".t-key").forEach((btn) => {
            btn.addEventListener("click", () => {
                const char = btn.dataset.char;
                if (!char) return;
                insertAtCursor(transcriptionInput, char);
                if (selectedCropIndex >= 0 && selectedCropIndex < imageAnnotations.length) {
                    imageAnnotations[selectedCropIndex].text = transcriptionInput.value;
                    renderAnnotationCards();
                }
            });
        });
    }

    function insertAtCursor(input, text) {
        input.focus();
        const start = input.selectionStart || 0;
        const end = input.selectionEnd || 0;
        const value = input.value;
        input.value = value.substring(0, start) + text + value.substring(end);
        input.selectionStart = input.selectionEnd = start + text.length;
        input.dispatchEvent(new Event("input"));
    }

    function setupFieldChips() {
        document.querySelectorAll(".chip-btn").forEach((chip) => {
            chip.addEventListener("click", () => {
                if (selectedCropIndex < 0 || selectedCropIndex >= imageAnnotations.length) {
                    setSaveStatus("Draw or select a crop first to assign this field.", "normal");
                    return;
                }
                const field = chip.dataset.field;
                const type = chip.dataset.type || "printed";
                fieldNameSelect.value = field;
                fieldNameCustom.style.display = "none";
                setSegmentedType(type);

                const ann = imageAnnotations[selectedCropIndex];
                ann.field_name = field;
                ann.type = type;
                if (type === "digits") ann.language = "digits";

                activeCropTitle.textContent = field;
                renderAnnotationCards();
                drawCanvas();
                transcriptionInput.focus();
            });
        });
    }

    // ==========================================
    // EVENT LISTENERS & SHORTCUTS
    // ==========================================
    function setupEventListeners() {
        // Image inputs
        imageInput.addEventListener("change", (e) => {
            if (e.target.files && e.target.files[0]) loadImageFromFile(e.target.files[0]);
        });
        emptyImageInput.addEventListener("change", (e) => {
            if (e.target.files && e.target.files[0]) loadImageFromFile(e.target.files[0]);
        });

        // Gallery Modal
        btnGalleryPick.addEventListener("click", openGalleryModal);
        btnEmptyGallery.addEventListener("click", openGalleryModal);
        modalCloseBtn.addEventListener("click", closeGalleryModal);
        galleryModal.addEventListener("click", (e) => {
            if (e.target === galleryModal) closeGalleryModal();
        });
        gallerySearch.addEventListener("input", filterGalleryGrid);

        // Rotation
        btnRotateCw.addEventListener("click", () => rotateImage(90));
        btnRotateCcw.addEventListener("click", () => rotateImage(-90));
        btnRotate180.addEventListener("click", () => rotateImage(180));

        // Tools & Zoom
        btnToolDraw.addEventListener("click", () => setToolMode("draw"));
        btnToolPan.addEventListener("click", () => setToolMode("pan"));
        btnZoomIn.addEventListener("click", () => zoomBy(1.25));
        btnZoomOut.addEventListener("click", () => zoomBy(0.8));
        btnZoomFit.addEventListener("click", zoomToFit);
        btnZoomReset.addEventListener("click", () => {
            zoom = 1.0;
            panOffset = { x: 0, y: 0 };
            applyTransform();
        });

        // Precision Tools
        btnTightenInk.addEventListener("click", tightenActiveBoxToInk);
        btnContrastBoost.addEventListener("click", () => {
            contrastBoost = !contrastBoost;
            btnContrastBoost.classList.toggle("active", contrastBoost);
            drawCanvas();
            updateInspector();
        });
        btnDeleteCrop.addEventListener("click", () => {
            if (selectedCropIndex >= 0) deleteCrop(selectedCropIndex);
        });

        // Inspector Form inputs
        cropTypeControl.querySelectorAll(".segment-btn").forEach((btn) => {
            btn.addEventListener("click", () => {
                const type = btn.dataset.type;
                setSegmentedType(type);
                if (selectedCropIndex >= 0 && selectedCropIndex < imageAnnotations.length) {
                    imageAnnotations[selectedCropIndex].type = type;
                    if (type === "digits") {
                        imageAnnotations[selectedCropIndex].language = "digits";
                        languageSelect.value = "digits";
                    }
                    renderAnnotationCards();
                    drawCanvas();
                }
            });
        });

        fieldNameSelect.addEventListener("change", () => {
            const val = fieldNameSelect.value;
            if (val === "CUSTOM") {
                fieldNameCustom.style.display = "block";
                fieldNameCustom.focus();
            } else {
                fieldNameCustom.style.display = "none";
                if (selectedCropIndex >= 0 && selectedCropIndex < imageAnnotations.length) {
                    imageAnnotations[selectedCropIndex].field_name = val;
                    activeCropTitle.textContent = val || "Unassigned";
                    renderAnnotationCards();
                    drawCanvas();
                }
            }
        });

        fieldNameCustom.addEventListener("input", () => {
            if (selectedCropIndex >= 0 && selectedCropIndex < imageAnnotations.length) {
                const val = fieldNameCustom.value.trim();
                imageAnnotations[selectedCropIndex].field_name = val;
                activeCropTitle.textContent = val || "Custom Field";
                renderAnnotationCards();
                drawCanvas();
            }
        });

        transcriptionInput.addEventListener("input", () => {
            if (selectedCropIndex >= 0 && selectedCropIndex < imageAnnotations.length) {
                imageAnnotations[selectedCropIndex].text = transcriptionInput.value;
                renderAnnotationCards();
            }
        });

        translationInput.addEventListener("input", () => {
            if (selectedCropIndex >= 0 && selectedCropIndex < imageAnnotations.length) {
                imageAnnotations[selectedCropIndex].translated_text = translationInput.value;
            }
        });

        languageSelect.addEventListener("change", () => {
            if (selectedCropIndex >= 0 && selectedCropIndex < imageAnnotations.length) {
                imageAnnotations[selectedCropIndex].language = languageSelect.value;
            }
        });

        // Layout Apply
        applyLayoutButton.addEventListener("click", applyCanonicalTemplate);

        // Save
        saveButton.addEventListener("click", saveAnnotations);

        // Keyboard Shortcuts
        window.addEventListener("keydown", (e) => {
            // Ignore if typing in inputs
            const activeTag = document.activeElement ? document.activeElement.tagName.toLowerCase() : "";
            const isTyping = activeTag === "input" || activeTag === "textarea" || activeTag === "select";

            if (e.code === "Space" && !isTyping) {
                isSpacePressed = true;
                viewport.classList.add("mode-pan");
                e.preventDefault();
            }

            if (!isTyping) {
                if (e.key === "Delete" || e.key === "Backspace") {
                    if (selectedCropIndex >= 0) {
                        deleteCrop(selectedCropIndex);
                        e.preventDefault();
                    }
                }
                if (e.key === "b" || e.key === "B") setToolMode("draw");
                if (e.key === "h" || e.key === "H") setToolMode("pan");
                if (e.key === "+" || e.key === "=") zoomBy(1.2);
                if (e.key === "-" || e.key === "_") zoomBy(0.8);

                // Nudge selected crop with Arrow keys
                if (selectedCropIndex >= 0 && ["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"].includes(e.key)) {
                    e.preventDefault();
                    const step = e.shiftKey ? 5 : 1;
                    const bbox = imageAnnotations[selectedCropIndex].bbox;
                    if (e.key === "ArrowUp") bbox[1] = Math.max(0, bbox[1] - step);
                    if (e.key === "ArrowDown") bbox[1] = Math.min(canvas.height - bbox[3], bbox[1] + step);
                    if (e.key === "ArrowLeft") bbox[0] = Math.max(0, bbox[0] - step);
                    if (e.key === "ArrowRight") bbox[0] = Math.min(canvas.width - bbox[2], bbox[0] + step);
                    drawCanvas();
                    updateInspector();
                }
            }
        });

        window.addEventListener("keyup", (e) => {
            if (e.code === "Space") {
                isSpacePressed = false;
                if (toolMode !== "pan") viewport.classList.remove("mode-pan");
            }
        });
    }

    function setToolMode(mode) {
        toolMode = mode;
        btnToolDraw.classList.toggle("active", mode === "draw");
        btnToolPan.classList.toggle("active", mode === "pan");
        viewport.classList.toggle("mode-pan", mode === "pan");
    }

    // ==========================================
    // GALLERY MODAL (32 EXISTING BILLS)
    // ==========================================
    function openGalleryModal() {
        galleryModal.classList.add("active");
        galleryModal.setAttribute("aria-hidden", "false");
        renderGalleryGrid(availableBills);
    }

    function closeGalleryModal() {
        galleryModal.classList.remove("active");
        galleryModal.setAttribute("aria-hidden", "true");
    }

    function renderGalleryGrid(bills) {
        galleryGrid.replaceChildren();
        if (!bills || bills.length === 0) {
            galleryGrid.innerHTML = "<p class='empty-list-hint'>No bills found in uploads directory.</p>";
            return;
        }

        bills.forEach((bill) => {
            const item = document.createElement("div");
            item.className = "gallery-item";
            item.title = `Click to load ${bill.filename}`;

            const thumbWrap = document.createElement("div");
            thumbWrap.className = "gallery-thumb-wrap";
            const img = document.createElement("img");
            img.src = `/api/uploads/raw/${encodeURIComponent(bill.filename)}`;
            img.alt = bill.filename;
            img.loading = "lazy";
            thumbWrap.appendChild(img);

            const title = document.createElement("div");
            title.className = "gallery-item-title";
            title.textContent = bill.filename;

            const size = document.createElement("div");
            size.className = "gallery-item-size";
            size.textContent = `${Math.round(bill.size_bytes / 1024)} KB`;

            item.append(thumbWrap, title, size);
            item.addEventListener("click", () => loadImageFromExisting(bill.filename));
            galleryGrid.appendChild(item);
        });
    }

    function filterGalleryGrid() {
        const query = gallerySearch.value.trim().toLowerCase();
        const filtered = availableBills.filter((b) => b.filename.toLowerCase().includes(query));
        renderGalleryGrid(filtered);
    }

    // ==========================================
    // APPLY CANONICAL TEMPLATE
    // ==========================================
    async function applyCanonicalTemplate() {
        if (!sourceImage) return;
        applyLayoutButton.disabled = true;
        setSaveStatus("Loading canonical crop layout...", "normal");
        try {
            const res = await fetch("/api/annotation-template");
            const result = await res.json();
            if (!res.ok || !result.success || !result.layout || result.layout.length === 0) {
                throw new Error(result.error || "No saved crop layout available yet.");
            }

            imageAnnotations = result.layout.map((item) => {
                const [nx, ny, nw, nh] = item.bbox_normalized;
                return {
                    field_name: item.field_name || "",
                    field_category: item.field_category || "custom",
                    text: "",
                    translated_text: "",
                    type: item.type || "printed",
                    language: item.language || "te",
                    bbox: [
                        Math.round(nx * canvas.width),
                        Math.round(ny * canvas.height),
                        Math.max(10, Math.round(nw * canvas.width)),
                        Math.max(10, Math.round(nh * canvas.height)),
                    ],
                };
            });

            selectCrop(0);
            renderAnnotationCards();
            drawCanvas();
            setSaveStatus("Canonical layout applied. Review each box and verify transcriptions.", "success");
        } catch (err) {
            setSaveStatus(err.message, "error");
        } finally {
            applyLayoutButton.disabled = false;
        }
    }

    // ==========================================
    // SAVE ANNOTATIONS
    // ==========================================
    async function saveAnnotations() {
        if (!sourceImage || imageAnnotations.length === 0) return;

        // Strict verification
        const missingFields = imageAnnotations.filter((ann) => !ann.field_name.trim() || !ann.text.trim());
        if (missingFields.length > 0) {
            setSaveStatus(`Validation failed: ${missingFields.length} crop(s) are missing Field Name or Exact Transcription.`, "error");
            return;
        }

        saveButton.disabled = true;
        setSaveStatus("Saving upright crops, records, and ground-truth...", "normal");

        const formData = new FormData();
        if (selectedImageFile) {
            formData.append("image", selectedImageFile);
        } else if (existingFilename) {
            formData.append("existing_filename", existingFilename);
        }
        formData.append("rotation", String(rotation));
        formData.append("bill_group_id", billGroupIdInput.value.trim());
        formData.append("annotations", JSON.stringify(imageAnnotations));

        try {
            const res = await fetch("/api/annotations/save", {
                method: "POST",
                body: formData,
            });
            const result = await res.json();
            if (!res.ok || !result.success) {
                throw new Error(result.error || "Server could not save annotations.");
            }

            setSaveStatus(
                `✓ Successfully saved ${result.saved_crops} upright crops! Ground truth saved for auto-cropping reference (${result.image_width} × ${result.image_height} px).`,
                "success"
            );
            saveButton.disabled = false;
        } catch (err) {
            setSaveStatus(err.message, "error");
            saveButton.disabled = false;
        }
    }

    function setSaveStatus(msg, type = "normal") {
        saveStatus.textContent = msg;
        saveStatus.className = `save-status-msg ${type}`;
    }

    // Start on DOM ready
    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }
})();
