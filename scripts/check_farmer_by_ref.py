import cv2
from pathlib import Path
import json

auto_hw_dir = Path("annotations/reviewed/auto_handwritten")
records = [json.loads(p.read_text(encoding="utf-8")) for p in auto_hw_dir.glob("*.json")]

# Group farmer name crops by reference source
for ref in set(r["reference_source"] for r in records):
    matching = [r for r in records if r["reference_source"] == ref and r["field_name"] == "FARMER_NAME"]
    print(f"Ref: {ref} has {len(matching)} farmer name crops.")
    # Show first 3 crop paths
    for m in matching[:3]:
        print("  Crop:", m["crop_path"], "Source:", m["source_image"])
