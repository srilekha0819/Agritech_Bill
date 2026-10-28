import sys
import os
import json
import uuid
from pathlib import Path
import cv2
import numpy as np
from werkzeug.utils import secure_filename

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(Path(".").resolve()))

import config
from scripts.region_alignment import image_features
from scripts.auto_annotate_printed import load_manual_printed_references

UPLOADS_DIR = Path(config.UPLOADS_DIR)
CROPS_HW_DIR = Path("annotations/crops/handwritten")
AUTO_HW_DIR = Path("annotations/reviewed/auto_handwritten")
TEMPLATES_PATH = Path("annotations/templates/region_templates.json")

CROPS_HW_DIR.mkdir(parents=True, exist_ok=True)
AUTO_HW_DIR.mkdir(parents=True, exist_ok=True)

tpl = json.loads(TEMPLATES_PATH.read_text(encoding="utf-8"))
regions_by_fn = {src["filename"]: src["regions"] for src in tpl["manual_source_images"]}
references = load_manual_printed_references()

rotations = [
    (None, 0),
    (cv2.ROTATE_90_CLOCKWISE, 90),
    (cv2.ROTATE_180, 180),
    (cv2.ROTATE_90_COUNTERCLOCKWISE, 270)
]

def align_upright(img):
    best = None
    best_inliers = 0
    for rot_code, angle in rotations:
        rot_img = img if rot_code is None else cv2.rotate(img, rot_code)
        kp, des, norm = image_features(rot_img)
        if des is None or len(kp) < 10:
            continue
        for ref in references:
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
            det = np.linalg.det(H[:2, :2])
            trace = H[0, 0] + H[1, 1]
            if det <= 0.1 or trace <= 0.4:
                continue
            inliers = int(mask.ravel().sum())
            if inliers > best_inliers:
                best_inliers = inliers
                best = {
                    "angle": angle,
                    "rot_img": rot_img,
                    "ref": ref,
                    "inliers": inliers,
                    "H": H
                }
    return best

upload_images = sorted([
    p for p in UPLOADS_DIR.iterdir()
    if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
])

print(f"Re-cropping {len(upload_images)} uploads with strict upright orientation...")

# Remove previous auto_handwritten records
for f in AUTO_HW_DIR.glob("*.json"):
    f.unlink()

count = 0
for idx, img_path in enumerate(upload_images, 1):
    img = cv2.imread(str(img_path))
    if img is None:
        continue
    al = align_upright(img)
    if not al:
        print(f"[{idx}] Failed alignment: {img_path.name}")
        continue
    
    rot_img = al["rot_img"]
    angle = al["angle"]
    H = al["H"]
    ref_fn = al["ref"]["filename"]
    H_inv = np.linalg.inv(H)
    img_h, img_w = rot_img.shape[:2]
    safe_stem = Path(secure_filename(img_path.name)).stem
    all_regions = regions_by_fn.get(ref_fn, [])
    hw_digit_regions = [r for r in all_regions if r.get("text_type") in ("handwritten", "digits")]
    
    for f_idx, reg in enumerate(hw_digit_regions):
        x, y, w, h = reg["bbox_xywh"]
        ref_poly = np.float32([
            [x, y], [x + w, y], [x + w, y + h], [x, y + h]
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
            "crop_path": os.path.relpath(str(crop_out_path), str(Path(".").resolve())).replace("\\", "/"),
            "text": "",
            "type": text_type,
            "language": "digits" if text_type == "digits" else "te",
            "annotation_source": "auto_crop_homography",
            "status": "PENDING_TRANSCRIPTION",
            "rotation_applied": angle,
            "alignment_inliers": al["inliers"],
            "reference_source": ref_fn
        }
        (AUTO_HW_DIR / f"{ann_id}.json").write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
        count += 1

print(f"Successfully cropped {count} upright field crops across {len(upload_images)} uploads.")
