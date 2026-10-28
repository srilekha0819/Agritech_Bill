import sys
import cv2
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))
from scripts.region_alignment import image_features
from scripts.auto_annotate_printed import load_manual_printed_references

refs = load_manual_printed_references()
ref = refs[1] # upright_WhatsApp Image 2026-09-17 at 4.09.23 PM (1).jpeg
img_path = Path("uploads/WhatsApp Image 2026-09-17 at 4.09.14 PM.jpeg")
img = cv2.imread(str(img_path))

for rot_code, angle in [(None, 0), (cv2.ROTATE_180, 180)]:
    rot_img = img if rot_code is None else cv2.rotate(img, rot_code)
    kp, des, norm = image_features(rot_img)
    matcher = cv2.BFMatcher(norm)
    matches = matcher.knnMatch(des, ref["des"], k=2)
    good = [m for m, n in matches if len(matches) > 1 and m.distance < 0.75 * n.distance]
    src_pts = np.float32([kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([ref["kp"][m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    print(f"Angle {angle}: H matrix:")
    print(H)
    det = np.linalg.det(H[:2, :2])
    trace = H[0, 0] + H[1, 1]
    print(f"Angle {angle}: det(H_2x2)={det:.3f}, trace(H_2x2)={trace:.3f}")
