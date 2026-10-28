import sys
import json
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
tpl = json.load(open('annotations/templates/region_templates.json', encoding='utf-8'))
for src in tpl['manual_source_images']:
    print(f"=== {src['filename']} ({len(src['regions'])} regions) ===")
    for r in src['regions']:
        print(f"  {r.get('text_type', 'unknown'):<12} | {r.get('region_key', ''):<25} | {r.get('region_type', '')}")
