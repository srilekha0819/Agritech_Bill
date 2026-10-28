import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))
from scripts.group_visual_bills import groups

recs = [json.loads(p.read_text(encoding='utf-8')) for p in Path('annotations/reviewed/auto_handwritten').glob('*.json')]
slips_by_src = {r['source_image']: r['crop_path'] for r in recs if r.get('field_name') == 'SLIP_NO'}

for idx, g in enumerate(groups, 1):
    rep = g[0]
    slip_crop = slips_by_src.get(rep, 'N/A')
    print(f"Group {idx} ({len(g)} files, rep={rep}): slip_crop={slip_crop}")
