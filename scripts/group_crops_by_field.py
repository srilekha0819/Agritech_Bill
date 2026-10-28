import sys
import json
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

auto_hw_dir = Path("annotations/reviewed/auto_handwritten")
records = [json.loads(p.read_text(encoding="utf-8")) for p in auto_hw_dir.glob("*.json")]

# Group by field_name and print some sample crop paths
by_field = {}
for r in records:
    by_field.setdefault(r["field_name"], []).append(r)

for field, recs in sorted(by_field.items()):
    print(f"Field: {field} -> {len(recs)} crops. Sample crop: {recs[0]['crop_path']}")
