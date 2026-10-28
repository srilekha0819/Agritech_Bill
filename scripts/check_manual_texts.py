import sys
import json
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

man_dir = Path("annotations/reviewed/manual")
for p in sorted(man_dir.glob("*.json")):
    d = json.loads(p.read_text(encoding="utf-8"))
    t = d.get("type")
    if t in ("digits", "handwritten"):
        print(f"[{d.get('source_filename')}] {t} {d.get('field_name')} -> '{d.get('text')}' (id={d.get('id')[:8]})")
