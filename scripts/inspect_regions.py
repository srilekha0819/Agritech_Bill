import sys
import json
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

tpl = json.load(open('annotations/templates/region_templates.json', encoding='utf-8'))
for src in tpl['manual_source_images']:
    print('SOURCE:', src['filename'])
    for r in src['regions']:
        print(f"  {r['region_key']}: type={r.get('text_type')} ann_id={r.get('annotation_id')}")
