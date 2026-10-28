import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(".").resolve()))
sys.stdout.reconfigure(encoding='utf-8')
from scripts.group_visual_bills import groups

recs = [json.loads(p.read_text(encoding='utf-8')) for p in Path('annotations/reviewed/auto_handwritten').glob('*.json')]

for idx, g in enumerate(groups, 1):
    rep = g[0]
    rep_recs = [r for r in recs if r['source_image'] == rep and r['type'] == 'handwritten']
    print(f"\n--- Group {idx} ({len(g)} images, Rep: {rep[:35]}) ---")
    for r in rep_recs:
        print(f"  Field: {r['field_name']} -> crop: {r['crop_path']}")
