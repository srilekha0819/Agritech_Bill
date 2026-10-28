#!/usr/bin/env python
"""
auto_crop_handwritten.py
Automatically crops and prepares annotation records for all handwritten and digit fields
from all uploaded bills using SIFT homography alignment to manual reference templates.

Outputs:
  - Crop images -> annotations/crops/handwritten/
  - Auto annotation JSONs -> annotations/reviewed/auto_handwritten/
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
from scripts.auto_annotate_printed import load_manual_printed_references, align_to_best_reference

UPLOADS_DIR = Path(config.UPLOADS_DIR)
CROPS_HW_DIR = BASE_DIR / "annotations" / "crops" / "handwritten"
AUTO_HW_DIR = BASE_DIR / "annotations" / "reviewed" / "auto_handwritten"
TEMPLATES_PATH = BASE_DIR / "annotations" / "templates" / "region_templates.json"

CROPS_HW_DIR.mkdir(parents=True, exist_ok=True)
AUTO_HW_DIR.mkdir(parents=True, exist_ok=True)


def auto_crop_all_handwritten():
    print("=" * 70)
    print("Auto-Cropping Handwritten and Digit Fields from Uploaded Bills")
    print("=" * 70)

    # 1. Load templates
    if not TEMPLATES_PATH.exists():
        print(f"[ERROR] Template path not found: {TEMPLATES_PATH}")
        return

    tpl = json.loads(TEMPLATES_PATH.read_text(encoding="utf-8"))
    regions_by_fn = {src["filename"]: src["regions"] for src in tpl["manual_source_images"]}

    # 2. Load manual references for SIFT alignment
    references = load_manual_printed_references()
    if not references:
        print("[ERROR] No references loaded.")
        return

    upload_images = sorted([
        p for p in UPLOADS_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ])

    print(f"\nProcessing {len(upload_images)} bill images from uploads/...\n")

    total_hw = 0
    total_digits = 0
    records = []

    for idx, img_path in enumerate(upload_images, 1):
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            continue

        alignment = align_to_best_reference(img, references)
        if not alignment or alignment.get("inliers", 0) < 25:
            print(f"[{idx}/{len(upload_images)}] Low alignment: {img_path.name}")
            continue

        rot_img = alignment["rot_img"]
        angle = alignment["angle"]
        H = alignment["H"]
        ref = alignment["ref"]
        ref_fn = ref["filename"]
        try:
            H_inv = np.linalg.inv(H)
        except Exception:
            continue

        img_h, img_w = rot_img.shape[:2]
        safe_stem = Path(secure_filename(img_path.name)).stem
        all_regions = regions_by_fn.get(ref_fn, [])
        hw_digit_regions = [r for r in all_regions if r.get("text_type") in ("handwritten", "digits")]

        bill_count = 0
        for f_idx, reg in enumerate(hw_digit_regions):
            x, y, w, h = reg["bbox_xywh"]
            ref_poly = np.float32([
                [x, y],
                [x + w, y],
                [x + w, y + h],
                [x, y + h]
            ]).reshape(-1, 1, 2)

            mapped_poly = cv2.perspectiveTransform(ref_poly, H_inv).reshape(-1, 2)
            min_x = max(0, int(round(mapped_poly[:, 0].min())))
            min_y = max(0, int(round(mapped_poly[:, 1].min())))
            max_x = min(img_w, int(round(mapped_poly[:, 0].max())))
            max_y = min(img_h, int(round(mapped_poly[:, 1].max())))
            cw = max_x - min_x
            ch = max_y - min_y

            if cw < 10 or ch < 8:
                continue

            crop = rot_img[min_y:max_y, min_x:max_x]
            ann_id = uuid.uuid4().hex
            crop_filename = f"hw_{safe_stem}_{f_idx:02d}_{ann_id[:8]}.png"
            crop_out_path = CROPS_HW_DIR / crop_filename
            cv2.imwrite(str(crop_out_path), crop)

            text_type = reg.get("text_type", "handwritten")
            if text_type == "digits":
                total_digits += 1
            else:
                total_hw += 1

            rec = {
                "id": ann_id,
                "source_image": img_path.name,
                "source_filename": img_path.name,
                "bill_group_id": safe_stem,
                "image_width": img_w,
                "image_height": img_h,
                "bbox": [min_x, min_y, cw, ch],
                "field_name": reg.get("region_type", reg.get("region_key", "")),
                "field_key": reg.get("region_key", ""),
                "crop_path": os.path.relpath(str(crop_out_path), str(BASE_DIR)).replace("\\", "/"),
                "text": "",  # To be transcribed / answered
                "type": text_type,
                "language": "digits" if text_type == "digits" else "te",
                "annotation_source": "auto_crop_homography",
                "status": "PENDING_TRANSCRIPTION",
                "rotation_applied": angle,
                "alignment_inliers": alignment.get("inliers", 0),
                "reference_source": ref_fn
            }

            json_out_path = AUTO_HW_DIR / f"{ann_id}.json"
            json_out_path.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
            records.append(rec)
            bill_count += 1

        print(f"[{idx}/{len(upload_images)}] {img_path.name[:35]:35s} -> {bill_count} crops ({alignment['inliers']} inliers)")

    print("\n" + "=" * 70)
    print(f"Auto-Cropping Complete!")
    print(f"Total Digit Crops Generated: {total_digits}")
    print(f"Total Telugu Handwritten Crops Generated: {total_hw}")
    print(f"Total Crops Generated: {len(records)}")
    print(f"Saved to: {CROPS_HW_DIR}")
    print(f"Records to: {AUTO_HW_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    auto_crop_all_handwritten()
