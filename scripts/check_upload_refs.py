import sys
import json
from pathlib import Path
import cv2
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

# Compare all 35 images to the two references
ref1_path = Path("uploads/upright_upright_WhatsApp Image 2026-09-17 at 4.09.25 PM.jpeg")
ref2_path = Path("uploads/upright_WhatsApp Image 2026-09-17 at 4.09.23 PM (1).jpeg")

im1 = cv2.imread(str(ref1_path), cv2.IMREAD_GRAYSCALE)
im2 = cv2.imread(str(ref2_path), cv2.IMREAD_GRAYSCALE)

auto_hw_dir = Path("annotations/reviewed/auto_handwritten")
records = [json.loads(p.read_text(encoding="utf-8")) for p in auto_hw_dir.glob("*.json")]

# Find unique references mapped
ref_sources = set(r.get("reference_source") for r in records)
print("Reference sources mapped during SIFT homography:", ref_sources)

# Check reference source count per upload
upload_to_ref = {}
for r in records:
    upload_to_ref[r.get("source_image")] = r.get("reference_source")

for u, ref in sorted(upload_to_ref.items()):
    print(f"{u} -> mapped to ref: {ref}")
