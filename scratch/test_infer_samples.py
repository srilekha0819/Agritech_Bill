import os
import sys
import json
from pathlib import Path
import cv2
import numpy as np
import torch

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(os.path.abspath(".")))

import config
from models.ocr_crnn import CRNN
from models.vocab import OCRVocab
from preprocessing.ocr_transform import prepare_ocr_crop, normalize_illumination

device = config.DEVICE

# 1. Printed Model
ckpt_p = torch.load("saved_models/ocr_crnn.pth", map_location=device, weights_only=False)
vocab_p = OCRVocab()
vocab_p.load("saved_models/ocr_vocab.json")
model_p = CRNN(img_h=config.OCR_IMG_H, in_channels=1, num_classes=len(vocab_p),
               hidden_size=ckpt_p.get("hidden_size", config.OCR_HIDDEN_SIZE),
               num_lstm_layers=ckpt_p.get("num_layers", config.OCR_NUM_LAYERS), dropout=0.0).to(device)
model_p.load_state_dict(ckpt_p["model_state_dict"])
model_p.eval()

# 2. Digits Model
ckpt_d = torch.load("saved_models/ocr_digits.pth", map_location=device, weights_only=False)
vocab_d = OCRVocab()
vocab_d.load("saved_models/ocr_digits_vocab.json")
model_d = CRNN(img_h=32, in_channels=1, num_classes=len(vocab_d), hidden_size=128, num_lstm_layers=2, dropout=0.0).to(device)
model_d.load_state_dict(ckpt_d["model_state_dict"])
model_d.eval()

# 3. Handwritten Model
ckpt_h = torch.load("saved_models/ocr_handwritten.pth", map_location=device, weights_only=False)
vocab_h = OCRVocab()
vocab_h.load("saved_models/ocr_handwritten_vocab.json")
model_h = CRNN(img_h=32, in_channels=1, num_classes=len(vocab_h), hidden_size=128, num_lstm_layers=2, dropout=0.0).to(device)
model_h.load_state_dict(ckpt_h["model_state_dict"])
model_h.eval()

def infer_crop(model, vocab, canvas):
    norm = (canvas.astype(np.float32) / 127.5) - 1.0
    tensor = torch.from_numpy(norm).unsqueeze(0).unsqueeze(0).float().to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=2)
        dec = model.decode_greedy(logits, blank_idx=vocab.blank_idx)[0]
        text = vocab.decode(dec).strip()
        conf = float(probs.max(dim=2).values.mean().item())
    return text, round(conf, 3)

# Test Digits on 5 real crops
print("=== TESTING DIGITS ===")
d_files = list(Path("annotations/reviewed/digits").glob("*.json"))[:5]
for df in d_files:
    d = json.loads(df.read_text(encoding="utf-8"))
    gt = d.get("text", "")
    cp = d.get("crop_path", "")
    img = cv2.imread(cp)
    if img is None:
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    norm = normalize_illumination(gray)
    h, w = norm.shape[:2]
    scale_w = max(16, min(int(w * 32 / float(max(h, 1))), 128))
    res = cv2.resize(norm, (scale_w, 32), interpolation=cv2.INTER_AREA)
    canvas = np.full((32, 128), float(np.median(res)), dtype=np.uint8)
    pad = (128 - scale_w) // 2
    canvas[:, pad:pad+scale_w] = res
    pred, conf = infer_crop(model_d, vocab_d, canvas)
    print(f"GT: {gt:<15} | Pred: {pred:<15} | Conf: {conf} | Match: {gt == pred}")

# Test Handwritten on 5 real crops
print("\n=== TESTING HANDWRITTEN ===")
h_files = list(Path("annotations/reviewed/handwritten").glob("*.json"))[:5]
for hf in h_files:
    d = json.loads(hf.read_text(encoding="utf-8"))
    gt = d.get("text", "")
    cp = d.get("crop_path", "")
    img = cv2.imread(cp)
    if img is None:
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    norm = normalize_illumination(gray)
    h, w = norm.shape[:2]
    scale_w = max(24, min(int(w * 32 / float(max(h, 1))), 320))
    res = cv2.resize(norm, (scale_w, 32), interpolation=cv2.INTER_AREA)
    canvas = np.full((32, 320), float(np.median(res)), dtype=np.uint8)
    canvas[:, :scale_w] = res
    pred, conf = infer_crop(model_h, vocab_h, canvas)
    print(f"GT: {gt:<20} | Pred: {pred:<20} | Conf: {conf} | Match: {gt == pred}")
