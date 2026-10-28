const pageLimit = 25;
let currentOffset = 0;
let currentIndex = 0;
let queueItems = [];
let pendingCount = 0;

const queueStatus = document.getElementById("queue-status");
const approveButton = document.getElementById("approve-crop");
const rejectButton = document.getElementById("reject-crop");

async function loadQueue(offset = currentOffset, index = 0) {
    queueStatus.textContent = "Loading review queue...";
    approveButton.disabled = true;
    rejectButton.disabled = true;
    try {
        const response = await fetch(`/api/review/queue?offset=${offset}&limit=${pageLimit}`);
        const result = await response.json();
        if (!response.ok || !result.success) throw new Error(result.error || "Could not load the queue.");
        currentOffset = result.offset;
        queueItems = result.items;
        pendingCount = result.pending_count;
        if (queueItems.length === 0 && currentOffset > 0) {
            return loadQueue(Math.max(0, currentOffset - pageLimit), 0);
        }
        currentIndex = Math.max(0, Math.min(index, queueItems.length - 1));
        renderCurrentItem();
    } catch (error) {
        queueStatus.textContent = error.message;
        document.getElementById("item-position").textContent = "0 / 0";
    }
}

function reviewFileUrl(path) {
    return path ? `/api/review/file?path=${encodeURIComponent(path)}` : "";
}

function setImage(imageId, placeholderId, path) {
    const image = document.getElementById(imageId);
    const placeholder = document.getElementById(placeholderId);
    image.onload = () => {
        image.hidden = false;
        placeholder.hidden = true;
    };
    image.onerror = () => {
        image.hidden = true;
        placeholder.hidden = false;
        placeholder.textContent = "Image could not be loaded.";
    };
    if (path) {
        image.hidden = false;
        image.src = reviewFileUrl(path);
    } else {
        image.removeAttribute("src");
        image.hidden = true;
        placeholder.hidden = false;
        placeholder.textContent = "No image available for this proposal.";
    }
}

function renderCurrentItem() {
    const item = queueItems[currentIndex];
    const position = currentOffset + currentIndex;
    document.getElementById("item-position").textContent =
        queueItems.length ? `${position + 1} / ${pendingCount}` : "0 / 0";
    document.getElementById("previous-item").disabled = pendingCount === 0 || position === 0;
    document.getElementById("next-item").disabled = pendingCount === 0 || position >= pendingCount - 1;

    if (!item) {
        queueStatus.textContent = "No pending crops.";
        document.getElementById("source-title").textContent = "Review complete";
        document.getElementById("region-title").textContent = "No crop selected";
        document.getElementById("region-type").value = "";
        document.getElementById("bill-group-id").value = "";
        document.getElementById("review-text").value = "";
        document.getElementById("review-translation").value = "";
        document.getElementById("error-category").value = "";
        document.getElementById("review-warning").textContent = "";
        setImage("source-preview", "source-placeholder", null);
        setImage("crop-preview", "crop-placeholder", null);
        return;
    }

    queueStatus.textContent = `${pendingCount} crops await a review decision.`;
    document.getElementById("source-title").textContent = item.source_image || "Uploaded bill";
    document.getElementById("region-title").textContent = item.region_id || item.region_type || "Proposed region";
    document.getElementById("region-type").value = item.region_type || "";
    document.getElementById("bill-group-id").value = item.bill_group_id || "";
    document.getElementById("crop-status").textContent = item.status || "PENDING_REVIEW";
    document.getElementById("bbox-label").textContent = item.bbox_xyxy
        ? `xyxy ${item.bbox_xyxy.join(", ")} · ${item.image_width} x ${item.image_height}`
        : item.bbox_xywh
            ? `xywh ${item.bbox_xywh.join(", ")} · ${item.image_width} x ${item.image_height}`
            : "No box coordinates";
    document.getElementById("review-warning").textContent =
        item.status === "GROUP_ID_REQUIRED"
            ? "This crop was approved before bill-level grouping existed. Confirm or change the group ID; use the same ID for other photos of this same bill."
            : item.status === "LOW_CONFIDENCE"
            ? "Registration or template variation is low confidence. Approve only after carefully verifying this crop against the source."
            : item.status === "OUT_OF_BOUNDS"
                ? "This crop extends outside the source image. Reject it and create a corrected manual crop instead."
                : item.status === "INVALID_CROP"
                    ? "This crop is invalid or empty. Reject it and create a corrected manual crop instead."
                    : "Automatic crop proposals are not ground truth. Verify the crop and type the visible text yourself.";
    document.getElementById("review-text").value = item.text || "";
    document.getElementById("review-translation").value = item.translated_text || "";
    document.getElementById("error-category").value = "";
    document.getElementById("text-type").value = item.text_type || "printed";
    document.getElementById("language-code").value = item.language || "";
    setImage("source-preview", "source-placeholder", item.preview_path || item.source_path);
    setImage("crop-preview", "crop-placeholder", item.crop_path);
    approveButton.disabled = !item.crop_path || item.status === "OUT_OF_BOUNDS" || item.status === "INVALID_CROP";
    rejectButton.disabled = false;
}

async function saveDecision(decision) {
    const item = queueItems[currentIndex];
    if (!item) return;
    const text = document.getElementById("review-text").value.trim();
    if (decision === "APPROVED" && !text) {
        queueStatus.textContent = "Enter the exact visible transcription before approving.";
        return;
    }
    const errorCategory = document.getElementById("error-category").value;
    if (decision === "REJECTED" && !errorCategory) {
        queueStatus.textContent = "Choose an error category before rejecting this crop.";
        return;
    }

    approveButton.disabled = true;
    rejectButton.disabled = true;
    queueStatus.textContent = decision === "APPROVED" ? "Saving approved ground truth..." : "Saving rejection...";
    try {
        const response = await fetch("/api/review/decision", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                annotation_id: item.annotation_id,
                source_kind: item.source_kind,
                run_id: item.run_id,
                source_image_id: item.source_image_id,
                bill_group_id: document.getElementById("bill-group-id").value.trim(),
                decision,
                text,
                translated_text: document.getElementById("review-translation").value.trim(),
                error_category: errorCategory,
                text_type: document.getElementById("text-type").value,
                language: document.getElementById("language-code").value.trim(),
            }),
        });
        const result = await response.json();
        if (!response.ok || !result.success) throw new Error(result.error || "Could not save review decision.");
        await loadQueue(currentOffset, currentIndex);
    } catch (error) {
        queueStatus.textContent = error.message;
        approveButton.disabled = false;
        rejectButton.disabled = false;
    }
}

document.getElementById("previous-item").addEventListener("click", () => {
    const position = currentOffset + currentIndex;
    if (position <= 0) return;
    if (currentIndex > 0) {
        currentIndex -= 1;
        renderCurrentItem();
    } else {
        loadQueue(Math.max(0, currentOffset - pageLimit), pageLimit - 1);
    }
});

document.getElementById("next-item").addEventListener("click", () => {
    if (currentIndex < queueItems.length - 1) {
        currentIndex += 1;
        renderCurrentItem();
    } else if (currentOffset + queueItems.length < pendingCount) {
        loadQueue(currentOffset + pageLimit, 0);
    }
});

approveButton.addEventListener("click", () => saveDecision("APPROVED"));
rejectButton.addEventListener("click", () => saveDecision("REJECTED"));
loadQueue();
