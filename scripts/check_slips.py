import cv2
from pathlib import Path
import json

# Check the crops of SLIP_NO across all 35 images
auto_hw_dir = Path("annotations/reviewed/auto_handwritten")
records = [json.loads(p.read_text(encoding="utf-8")) for p in auto_hw_dir.glob("*.json")]

slip_crops = [r for r in records if r.get("field_name") == "SLIP_NO"]
print("Total SLIP_NO crops:", len(slip_crops))

# Check reference sources
for r in slip_crops[:10]:
    print(r["source_image"], "-> ref:", r.get("reference_source"), "crop:", r.get("crop_path"))
