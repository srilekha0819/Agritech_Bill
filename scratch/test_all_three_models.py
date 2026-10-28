import os
import sys
import json
import cv2
import numpy as np
import torch

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(os.path.abspath(".")))

import config
from models.ocr_crnn import CRNN
from models.vocab import OCRVocab

device = config.DEVICE
print(f"Testing on device: {device}")

# 1. Test Printed Model
print("\n--- 1. Testing Printed Model ---")
ckpt_p = torch.load("saved_models/ocr_crnn.pth", map_location=device, weights_only=False)
vocab_p = OCRVocab()
vocab_p.load("saved_models/ocr_vocab.json")
model_p = CRNN(
    img_h=config.OCR_IMG_H,
    in_channels=config.OCR_IN_CHANNELS,
    num_classes=len(vocab_p),
    hidden_size=ckpt_p.get("hidden_size", config.OCR_HIDDEN_SIZE),
    num_lstm_layers=ckpt_p.get("num_layers", config.OCR_NUM_LAYERS),
    dropout=0.0
).to(device)
model_p.load_state_dict(ckpt_p["model_state_dict"])
model_p.eval()
print(f"Printed model loaded. Classes: {len(vocab_p)}, CER: {ckpt_p.get('validation_cer')}")

# 2. Test Digits Model
print("\n--- 2. Testing Digits Model ---")
ckpt_d = torch.load("saved_models/ocr_digits.pth", map_location=device, weights_only=False)
vocab_d = OCRVocab()
vocab_d.load("saved_models/ocr_digits_vocab.json")
model_d = CRNN(
    img_h=32,
    in_channels=1,
    num_classes=len(vocab_d),
    hidden_size=128,
    num_lstm_layers=2,
    dropout=0.0
).to(device)
model_d.load_state_dict(ckpt_d["model_state_dict"])
model_d.eval()
print(f"Digits model loaded. Classes: {len(vocab_d)}, Acc: {ckpt_d.get('exact_match_acc')}")

# 3. Test Handwritten Model
print("\n--- 3. Testing Handwritten Model ---")
ckpt_h = torch.load("saved_models/ocr_handwritten.pth", map_location=device, weights_only=False)
vocab_h = OCRVocab()
vocab_h.load("saved_models/ocr_handwritten_vocab.json")
model_h = CRNN(
    img_h=32,
    in_channels=1,
    num_classes=len(vocab_h),
    hidden_size=128,
    num_lstm_layers=2,
    dropout=0.0
).to(device)
model_h.load_state_dict(ckpt_h["model_state_dict"])
model_h.eval()
print(f"Handwritten model loaded. Classes: {len(vocab_h)}, CER: {ckpt_h.get('validation_cer')}")

print("\nAll 3 models loaded successfully!")
