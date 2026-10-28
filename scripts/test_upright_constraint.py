import sys
import cv2
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))
from scripts.region_alignment import image_features
from scripts.auto_annotate_printed import load_manual_printed_references

refs = load_manual_printed_references()
uploads = sorted([
    p for p in Path("uploads").iterdir()
    if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
])

print(f"Testing upright orientation validation across {len(uploads)} images:")

rotations = [
    (None, 0),
    (cv2.ROTATE_90_CLOCKWISE, 90),
    (cv2.ROTATE_180, 180),
    (cv2.ROTATE_90_COUNTERCLOCKWISE, 270)
]

for idx, p in enumerate(uploads, 1):
    img = cv2.imread(str(p))
    best = None
    best_inliers = 0
    
    for rot_code, angle in rotations:
        rot_img = img if rot_code is None else cv2.rotate(img, rot_code)
        kp, des, norm = image_features(rot_img)
        if des is None or len(kp) < 10:
            continue
        for ref in refs:
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
            # Orientation check: H must preserve orientation (positive trace and det)
            det = np.linalg.det(H[:2, :2])
            trace = H[0, 0] + H[1, 1]
            if det <= 0.1 or trace <= 0.2:
                # Upside down or mirrored match! Skip!
                continue
            inliers = int(mask.ravel().sum())
            if inliers > best_inliers:
                best_inliers = inliers
                best = {"angle": angle, "ref": ref["filename"], "inliers": inliers, "det": det, "trace": trace}
    
    if best:
        print(f"[{idx:02d}] {p.name[:35]}: angle={best['angle']} (inliers={best['inliers']}, trace={best['trace']:.2f}) -> {best['ref'][:25]}")
    else:
        print(f"[{idx:02d}] {p.name[:35]}: NO UPRIGHT MATCH FOUND")
