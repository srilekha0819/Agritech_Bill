#!/usr/bin/env python
"""
auto_annotate_printed.py
Annotates and crops printed fields from all uploaded bills using manual reference annotations.
Uses SIFT feature matching and RANSAC homography to transfer printed bounding boxes.
Saves crops to annotations/crops/ and annotation records to annotations/reviewed/auto/.
"""
import os
import sys
import json
import uuid
from pathlib import Path
from collections import Counter

import cv2
import numpy as np
from werkzeug.utils import secure_filename

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

sys.stdout.reconfigure(encoding='utf-8')

import config
from scripts.region_alignment import image_features

UPLOADS_DIR = Path(config.UPLOADS_DIR)
MANUAL_ANNOTATIONS_DIR = BASE_DIR / "annotations" / "reviewed" / "manual"
AUTO_ANNOTATIONS_DIR = BASE_DIR / "annotations" / "reviewed" / "auto"
CROPS_DIR = BASE_DIR / "annotations" / "crops"
TEMPLATES_PATH = BASE_DIR / "annotations" / "templates" / "region_templates.json"

AUTO_ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
CROPS_DIR.mkdir(parents=True, exist_ok=True)


def load_manual_printed_references():
    """Load manual printed annotations grouped by source filename."""
    template = json.loads(TEMPLATES_PATH.read_text(encoding="utf-8"))
    manual_sources = template["manual_source_images"]

    printed_by_file = {}
    for jf in MANUAL_ANNOTATIONS_DIR.glob("*.json"):
        try:
            d = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            continue
        if d.get("type") == "printed":
            text = (d.get("text") or "").strip()
            # Ignore purely numeric or punctuation
            if not text or set(text) <= set("0123456789-_ ."):
                continue
            src = d.get("source_filename")
            if src not in printed_by_file:
                printed_by_file[src] = []
            printed_by_file[src].append(d)

    references = []
    for src in manual_sources:
        fn = src["filename"]
        img_path = UPLOADS_DIR / fn
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        kp, des, norm = image_features(img)
        anns = printed_by_file.get(fn, [])
        if not anns:
            continue
        references.append({
            "filename": fn,
            "image": img,
            "width": src["width"],
            "height": src["height"],
            "kp": kp,
            "des": des,
            "norm": norm,
            "annotations": anns
        })
        print(f"[INFO] Loaded reference '{fn}' with {len(anns)} printed annotations")

    return references


def align_to_best_reference(image: np.ndarray, references: list):
    """Test 4 rotations and match against all references to find best alignment."""
    best = None
    best_inliers = 0

    rotations = [
        (None, 0),
        (cv2.ROTATE_90_CLOCKWISE, 90),
        (cv2.ROTATE_180, 180),
        (cv2.ROTATE_90_COUNTERCLOCKWISE, 270)
    ]

    for rot_code, angle in rotations:
        rot_img = image if rot_code is None else cv2.rotate(image, rot_code)
        kp, des, norm = image_features(rot_img)
        if des is None or len(kp) < 10:
            continue

        for ref in references:
            if norm != ref["norm"] or ref["des"] is None:
                continue
            matcher = cv2.BFMatcher(norm)
            matches = matcher.knnMatch(des, ref["des"], k=2)
            good = [m for m, n in matches if len(matches) > 1 and m.distance < 0.75 * n.distance]
            if len(good) < 20:
                continue

            src_pts = np.float32([kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([ref["kp"][m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if H is None or mask is None:
                continue

            inliers = int(mask.ravel().sum())
            if inliers > best_inliers:
                best_inliers = inliers
                best = {
                    "angle": angle,
                    "rot_img": rot_img,
                    "ref": ref,
                    "inliers": inliers,
                    "inlier_ratio": inliers / max(1, len(good)),
                    "good_matches": len(good),
                    "H": H
                }

    return best


def auto_annotate_all():
    print("=" * 70)
    print("Starting Reference-Based Auto-Annotation for Printed Fields")
    print("=" * 70)

    references = load_manual_printed_references()
    if not references:
        print("[ERROR] No valid manual printed references found!")
        return

    reference_filenames = {r["filename"] for r in references}
    upload_images = sorted([
        p for p in UPLOADS_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        and p.name not in reference_filenames
    ])

    print(f"\nProcessing {len(upload_images)} unannotated bill images from uploads/...\n")

    total_crops_generated = 0
    total_bills_annotated = 0

    for idx, img_path in enumerate(upload_images, 1):
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[{idx}/{len(upload_images)}] Skipped (unreadable): {img_path.name}")
            continue

        alignment = align_to_best_reference(img, references)
        if not alignment or alignment["inliers"] < 25:
            inliers = alignment["inliers"] if alignment else 0
            print(f"[{idx}/{len(upload_images)}] Low alignment ({inliers} inliers): {img_path.name}")
            continue

        rot_img = alignment["rot_img"]
        H = alignment["H"]
        ref = alignment["ref"]
        H_ref_to_img = np.linalg.inv(H)
        img_h, img_w = rot_img.shape[:2]

        safe_stem_name = Path(secure_filename(img_path.name)).stem
        bill_crops = 0

        for f_idx, ann in enumerate(ref["annotations"]):
            x, y, w, h = ann["bbox"]
            ref_poly = np.float32([
                [x, y],
                [x + w, y],
                [x + w, y + h],
                [x, y + h]
            ]).reshape(-1, 1, 2)

            mapped_poly = cv2.perspectiveTransform(ref_poly, H_ref_to_img).reshape(-1, 2)

            # Check bounds
            min_x, min_y = mapped_poly.min(axis=0)
            max_x, max_y = mapped_poly.max(axis=0)
            if min_x < -20 or min_y < -20 or max_x > img_w + 20 or max_y > img_h + 20:
                continue

            dst_w, dst_h = max(8, int(round(w))), max(8, int(round(h)))
            dst_rect = np.float32([
                [0, 0],
                [dst_w - 1, 0],
                [dst_w - 1, dst_h - 1],
                [0, dst_h - 1]
            ])
            M = cv2.getPerspectiveTransform(mapped_poly.astype(np.float32), dst_rect)
            crop = cv2.warpPerspective(rot_img, M, (dst_w, dst_h), borderValue=(255, 255, 255))

            if crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 8:
                continue

            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            std = float(gray.std())
            if std < 4.0:
                # Blank / uniform crop (out of image or empty margin)
                continue

            # Save crop
            ann_id = uuid.uuid4().hex
            crop_filename = f"auto_{safe_stem_name}_{f_idx:02d}_{ann_id[:8]}.png"
            crop_path = CROPS_DIR / crop_filename
            cv2.imwrite(str(crop_path), crop)

            # Save annotation JSON
            ann_record = {
                "id": ann_id,
                "source_image": img_path.name,
                "source_filename": img_path.name,
                "bill_group_id": safe_stem_name,
                "image_width": img_w,
                "image_height": img_h,
                "bbox": [int(round(min_x)), int(round(min_y)), int(round(max_x - min_x)), int(round(max_y - min_y))],
                "field_name": ann.get("field_name", ""),
                "field_category": ann.get("field_category", "printed"),
                "crop_path": str(crop_path.relative_to(BASE_DIR)),
                "text": ann.get("text", "").strip(),
                "translated_text": ann.get("translated_text", "").strip(),
                "type": "printed",
                "language": ann.get("language", "te"),
                "annotation_source": "auto_reference_transfer",
                "status": "APPROVED",
                "rotation_applied": alignment["angle"],
                "alignment_inliers": alignment["inliers"],
                "reference_source": ref["filename"]
            }

            ann_file = AUTO_ANNOTATIONS_DIR / f"{ann_id}.json"
            ann_file.write_text(json.dumps(ann_record, ensure_ascii=False, indent=2), encoding="utf-8")

            bill_crops += 1
            total_crops_generated += 1

        total_bills_annotated += 1
        print(f"[{idx}/{len(upload_images)}] {img_path.name[:35]:35s} | rot={alignment['angle']:3d} | inliers={alignment['inliers']:4d} | crops={bill_crops:2d}")

    print("\n" + "=" * 70)
    print(f"Summary: Auto-annotated {total_bills_annotated} bills, generated {total_crops_generated} printed crops.")
    print(f"Auto-annotations saved in: {AUTO_ANNOTATIONS_DIR}")
    print(f"Crop images saved in: {CROPS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    auto_annotate_all()
