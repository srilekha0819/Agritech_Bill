import cv2
from pathlib import Path
import json

auto_hw_dir = Path("annotations/reviewed/auto_handwritten")
records = [json.loads(p.read_text(encoding="utf-8")) for p in auto_hw_dir.glob("*.json")]

# Let's inspect source_image names and reference_source
by_src = {}
for r in records:
    by_src.setdefault(r["source_image"], []).append(r)

print(f"Number of upload files: {len(by_src)}")
for src, recs in list(by_src.items())[:5]:
    ref = recs[0]["reference_source"]
    print(f"File: {src} -> Ref: {ref}")
