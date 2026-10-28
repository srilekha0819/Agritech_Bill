import os
from pathlib import Path
import cv2

uploads = sorted([
    p for p in Path("uploads").iterdir()
    if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
])

print(f"Total uploaded files: {len(uploads)}")
for idx, p in enumerate(uploads, 1):
    img = cv2.imread(str(p))
    h, w = img.shape[:2] if img is not None else (0, 0)
    print(f"[{idx:02d}] {p.name} (w={w}, h={h})")
