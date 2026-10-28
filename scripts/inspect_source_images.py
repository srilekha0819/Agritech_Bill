import sys
import json
from pathlib import Path
import cv2
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

auto_hw_dir = Path("annotations/reviewed/auto_handwritten")
records = [json.loads(p.read_text(encoding="utf-8")) for p in auto_hw_dir.glob("*.json")]

# Group by source image
by_img = {}
for r in records:
    src = r.get("source_image")
    by_img.setdefault(src, []).append(r)

print(f"Total images with auto-crops: {len(by_img)}")
for src, recs in sorted(by_img.items()):
    types = [r.get("type") for r in recs]
    fields = [r.get("field_name") for r in recs]
    print(f"{src}: {len(recs)} crops (digits: {types.count('digits')}, hw: {types.count('handwritten')})")
