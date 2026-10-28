import cv2
from pathlib import Path
import json

auto_hw_dir = Path("annotations/reviewed/auto_handwritten")
records = [json.loads(p.read_text(encoding="utf-8")) for p in auto_hw_dir.glob("*.json")]

# Let's inspect farmer_name crops
farmer_crops = [r for r in records if r.get("field_name") == "FARMER_NAME"]
print(f"Total farmer crops: {len(farmer_crops)}")
for r in farmer_crops[:10]:
    print(r["source_image"], "-> ref:", r.get("reference_source"))
