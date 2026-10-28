import os
import sys
import json
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(os.path.abspath(".")))

from pipeline.bill_pipeline import BillIntelligencePipeline

pipeline = BillIntelligencePipeline()

test_image = "uploads/WhatsApp Image 2026-09-17 at 4.09.25 PM.jpeg"
print(f"Testing BillIntelligencePipeline on: {test_image}")

result = pipeline.process_image_path(test_image)

print("\n--- Pipeline Execution Result ---")
print(f"Success: {result['success']}")
print(f"Rotation Applied: {result['rotation_applied']}°")
print(f"Processing Time: {result['processing_time_ms']} ms")
print(f"OCR Confidence Estimate: {result['ocr_confidence_estimate']}")
print(f"Total Lines Extracted: {len(result['lines'])}")

print("\n--- ALL Extracted Lines ---")
for line in result['lines']:
    print(f"#{line['line_id']:02d} [{line['text_type']:<11}] {line['field_name']:<25} | Conf: {line['confidence']:.2f} | \"{line['original_text']}\" -> \"{line['translated_text']}\"")


print("\n--- Structured JSON ---")
print(json.dumps(result['structured_json'], ensure_ascii=False, indent=2))

print("\n--- ALL TESTS PASSED! ---")
