import json
from pathlib import Path

recs = [json.loads(p.read_text(encoding='utf-8')) for p in Path('annotations/reviewed/auto_handwritten').glob('*.json')]
slips = [r for r in recs if r.get('field_name') == 'SLIP_NO']
slips.sort(key=lambda x: x['source_image'])

print(f"Total SLIP_NO crops: {len(slips)}")
for s in slips:
    print(f"{s['source_image']} -> {s['crop_path']}")
