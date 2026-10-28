import cv2
from pathlib import Path
import json
import numpy as np

auto_hw_dir = Path("annotations/reviewed/auto_handwritten")
records = [json.loads(p.read_text(encoding="utf-8")) for p in auto_hw_dir.glob("*.json")]

# Let's inspect the crops for COMMODITY and FARMER_NAME
fn_crops = [r for r in records if r.get("field_name") == "FARMER_NAME"]
ref_23_fn = "annotations/crops/manual_374f6efbf5574d8fa4e78d599006a2e5.png" # T.నరసింహారావు
ref_25_fn = "annotations/crops/manual_24a24bf2cc314e81b4c579700d55dc5a.png" # M.వినీల్

# Let's see if crops exist
print("Checking crops...")
for r in fn_crops:
    cp = Path(r["crop_path"])
    if not cp.exists():
        print("Missing:", cp)
print("All farmer crops exist:", all(Path(r["crop_path"]).exists() for r in fn_crops))
