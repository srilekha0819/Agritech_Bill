import sys
import cv2
from pathlib import Path

sys.path.insert(0, str(Path(".").resolve()))
sys.stdout.reconfigure(encoding='utf-8')

from pipeline.ocr_engine import PrintedTextOCREngine

engine = PrintedTextOCREngine()
print("Engine ready:", engine.is_ready)

img_path = "uploads/upright_WhatsApp Image 2026-09-17 at 4.09.23 PM (1).jpeg"
im = cv2.imread(img_path)
res = engine.process_image(im)
print(f"Status: {res['model_status']}, rotation: {res['rotation_applied']}, lines: {len(res['lines'])}")
for i, line in enumerate(res['lines']):
    print(f"[{i+1:02d}] {line['field_name']:25s} -> '{line['text']}' (conf: {line['confidence']})")
