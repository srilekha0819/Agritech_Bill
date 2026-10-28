import cv2
import numpy as np
import torch
from typing import List, Tuple

import config


def _segment_lines_in_crop(gray: np.ndarray) -> List[np.ndarray]:
    """
    Segment an already-horizontal crop into individual text-line sub-images.
    Uses morphological row projection to find line boundaries.
    Returns list of (sub-image arrays), each containing one text line.
    Falls back to the whole image if segmentation finds nothing useful.
    """
    h, w = gray.shape[:2]
    if h < 24 or w < 16:
        return [gray]

    # Normalize illumination using background division
    ksize = max(int(min(h, w) * 0.08) | 1, 11)
    dilated = cv2.dilate(gray, np.ones((5, 5), np.uint8))
    bg = cv2.medianBlur(dilated, ksize)
    bg_safe = np.maximum(bg.astype(np.float32), 1.0)
    norm = np.clip((gray.astype(np.float32) / bg_safe) * 255.0, 0, 255).astype(np.uint8)

    # Binarize: ink = white (255)
    _, binary = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Remove long horizontal/vertical rules to avoid false row merges
    if w >= 30:
        hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(w // 4, 20), 1))
        h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, hk)
        binary = cv2.subtract(binary, h_lines)

    # Horizontal projection: count ink pixels per row
    row_proj = np.sum(binary > 0, axis=1)  # shape (h,)

    # Smooth with a small window to bridge inter-character gaps within a line
    kernel_size = max(3, h // 20)
    smoothed = np.convolve(row_proj, np.ones(kernel_size) / kernel_size, mode="same")

    # Adaptive threshold: rows with ink above 2% of image width
    ink_threshold = max(w * 0.02, 2.0)
    ink_rows = smoothed > ink_threshold

    # Find contiguous runs of ink rows (these are the text lines)
    line_segments: List[Tuple[int, int]] = []
    in_line = False
    seg_start = 0
    padding = max(2, h // 40)

    for row_idx in range(h):
        if ink_rows[row_idx] and not in_line:
            in_line = True
            seg_start = row_idx
        elif not ink_rows[row_idx] and in_line:
            in_line = False
            y1 = max(0, seg_start - padding)
            y2 = min(h, row_idx + padding)
            if (y2 - y1) >= 8:  # minimum line height
                line_segments.append((y1, y2))

    # Close the last segment
    if in_line:
        y1 = max(0, seg_start - padding)
        y2 = h
        if (y2 - y1) >= 8:
            line_segments.append((y1, y2))

    # Merge segments that are too close together (gap < median line height / 3)
    if len(line_segments) > 1:
        heights = [y2 - y1 for y1, y2 in line_segments]
        median_h = float(np.median(heights))
        min_gap = max(int(median_h / 3), 4)

        merged: List[Tuple[int, int]] = [line_segments[0]]
        for seg in line_segments[1:]:
            prev = merged[-1]
            gap = seg[0] - prev[1]
            if gap <= min_gap:
                merged[-1] = (prev[0], seg[1])
            else:
                merged.append(seg)
        line_segments = merged

    # Filter out segments that are too short (< 8px height)
    line_segments = [(y1, y2) for y1, y2 in line_segments if (y2 - y1) >= 8]

    if not line_segments:
        return [gray]

    # Return sub-crops for each line
    return [gray[y1:y2, :] for y1, y2 in line_segments]


def normalize_illumination(gray: np.ndarray) -> np.ndarray:
    """Background-subtraction illumination normalization."""
    ksize = max(int(min(gray.shape) * 0.05) | 1, 11)
    dilated = cv2.dilate(gray, np.ones((5, 5), np.uint8))
    bg = cv2.medianBlur(dilated, ksize)
    bg_safe = np.maximum(bg.astype(np.float32), 1.0)
    return np.clip((gray.astype(np.float32) / bg_safe) * 255.0, 0, 255).astype(np.uint8)


def prepare_ocr_crop(crop: np.ndarray) -> torch.Tensor:
    """
    Apply the shared CRNN training/inference transform.
    Matches train_printed.py prepare_crop 100% mathematically.
    """
    if crop is None or crop.size == 0:
        raise ValueError("OCR crop must contain image pixels.")
    if crop.ndim == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop.copy()

    # If crop is vertical (from sideways bill photos), rotate 90 CCW to make it horizontal
    if gray.shape[0] > int(gray.shape[1] * 1.15):
        gray = cv2.rotate(gray, cv2.ROTATE_90_COUNTERCLOCKWISE)

    norm = normalize_illumination(gray)
    h, w = norm.shape[:2]
    th = config.OCR_IMG_H
    tw = config.OCR_IMG_W
    scale_w = max(16, min(int(w * th / float(max(h, 1))), tw))
    resized = cv2.resize(norm, (scale_w, th), interpolation=cv2.INTER_AREA)
    bg_val = float(np.median(resized))
    canvas = np.full((th, tw), bg_val, dtype=np.uint8)
    canvas[:, :scale_w] = resized
    normalized = (canvas.astype(np.float32) / 127.5) - 1.0
    return torch.from_numpy(normalized).unsqueeze(0).float()


def prepare_digit_crop(crop: np.ndarray) -> torch.Tensor:
    """
    Digit-specific transform with dense 128px canvas matching train_digits_v2.py.
    Eliminates dead background padding so CTC timesteps are concentrated on numbers.
    """
    if crop is None or crop.size == 0:
        raise ValueError("Digit crop must contain image pixels.")
    if crop.ndim == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop.copy()

    if gray.shape[0] > int(gray.shape[1] * 1.15):
        gray = cv2.rotate(gray, cv2.ROTATE_90_COUNTERCLOCKWISE)

    norm = normalize_illumination(gray)
    th = config.OCR_IMG_H  # 32
    tw = 128               # 128px dense canvas matching train_digits_v2.py
    resized = cv2.resize(norm, (tw, th), interpolation=cv2.INTER_AREA)
    normalized = (resized.astype(np.float32) / 127.5) - 1.0
    return torch.from_numpy(normalized).unsqueeze(0).float()


def split_vertical_crop_into_lines(crop: np.ndarray) -> List[np.ndarray]:

    """
    For a vertical crop (tall-and-narrow strip from a sideways bill image),
    rotates it to horizontal and then segments it into individual text lines.
    Returns a list of horizontal text-line crops (BGR or gray, matching input).

    This is the KEY function used in training to correctly split multi-line
    vertical annotation crops into individual lines the CRNN can handle.
    """
    if crop is None or crop.size == 0:
        return []

    is_color = crop.ndim == 3

    if crop.ndim == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop.copy()

    h, w = gray.shape[:2]

    # Only rotate if genuinely vertical
    if h > w * 1.15:
        gray_rotated = cv2.rotate(gray, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if is_color:
            color_rotated = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            color_rotated = gray_rotated
    else:
        gray_rotated = gray
        color_rotated = crop

    # Segment into lines on the rotated (now horizontal) image
    line_gray_crops = _segment_lines_in_crop(gray_rotated)

    if len(line_gray_crops) <= 1:
        # No meaningful segmentation — return the full rotated crop
        return [color_rotated]

    # Map line y-boundaries back to color crop if needed
    if is_color:
        # Re-segment to get y-bounds
        result = []
        rh = gray_rotated.shape[0]
        prev_y = 0
        for line_crop in line_gray_crops:
            lh = line_crop.shape[0]
            y1 = prev_y
            y2 = min(rh, prev_y + lh)
            # Add padding to match what _segment_lines_in_crop added
            result.append(color_rotated[y1:y2, :])
            prev_y = y2
        return result
    else:
        return line_gray_crops

