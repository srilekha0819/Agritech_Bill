import sys
import cv2
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))
from scripts.region_alignment import image_features
from scripts.auto_annotate_printed import load_manual_printed_references

refs = load_manual_printed_references()
img_path = Path("uploads/WhatsApp Image 2026-09-17 at 4.09.14 PM.jpeg")
img = cv2.imread(str(img_path))

rotations = [
    (None, 0),
    (cv2.ROTATE_90_CLOCKWISE, 90),
    (cv2.ROTATE_180, 180),
    (cv2.ROTATE_90_COUNTERCLOCKWISE, 270)
]

for rot_code, angle in rotations:
    rot_img = img if rot_code is None else cv2.rotate(img, rot_code)
    kp, des, norm = image_features(rot_img)
    print(f"Angle {angle}: {len(kp)} keypoints")
    for ref in refs:
        matcher = cv2.BFMatcher(norm)
        matches = matcher.knnMatch(des, ref["des"], k=2)
        good = [m for m, n in matches if len(matches) > 1 and m.distance < 0.75 * n.distance]
        if len(good) < 20:
            print(f"  Ref {ref['filename'][:30]}: good matches = {len(good)}")
            continue
        src_pts = np.float32([kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([ref["kp"][m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        inliers = int(mask.ravel().sum()) if mask is not None else 0
        print(f"  Ref {ref['filename'][:30]}: good={len(good)}, inliers={inliers}")
