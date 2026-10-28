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
from preprocessing.ocr_transform import prepare_ocr_crop
from scripts.region_alignment import image_features

device = config.DEVICE

# 1. Load Printed Model
ckpt_p = torch.load("saved_models/ocr_crnn.pth", map_location=device, weights_only=False)
vocab_p = OCRVocab()
vocab_p.load("saved_models/ocr_vocab.json")
model_p = CRNN(img_h=32, in_channels=1, num_classes=len(vocab_p),
               hidden_size=ckpt_p.get("hidden_size", 192),
               num_lstm_layers=ckpt_p.get("num_layers", 2), dropout=0.0).to(device)
model_p.load_state_dict(ckpt_p["model_state_dict"])
model_p.eval()

# 2. Load Digits Model
ckpt_d = torch.load("saved_models/ocr_digits.pth", map_location=device, weights_only=False)
vocab_d = OCRVocab()
vocab_d.load("saved_models/ocr_digits_vocab.json")
model_d = CRNN(img_h=32, in_channels=1, num_classes=len(vocab_d),
               hidden_size=ckpt_d.get("hidden_size", 192),
               num_lstm_layers=ckpt_d.get("num_layers", 2), dropout=0.0).to(device)
model_d.load_state_dict(ckpt_d["model_state_dict"])
model_d.eval()

# 3. Load Handwritten Model
ckpt_h = torch.load("saved_models/ocr_handwritten.pth", map_location=device, weights_only=False)
vocab_h = OCRVocab()
vocab_h.load("saved_models/ocr_handwritten_vocab.json")
model_h = CRNN(img_h=32, in_channels=1, num_classes=len(vocab_h),
               hidden_size=ckpt_h.get("hidden_size", 192),
               num_lstm_layers=ckpt_h.get("num_layers", 2), dropout=0.0).to(device)
model_h.load_state_dict(ckpt_h["model_state_dict"])
model_h.eval()

def recognize_crop(crop: np.ndarray, model: CRNN, vocab: OCRVocab):
    if crop is None or crop.size == 0:
        return "", 0.0
    h, w = crop.shape[:2]
    if h < 8 or w < 10:
        return "", 0.0
    try:
        tensor = prepare_ocr_crop(crop).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=2)
            dec = model.decode_greedy(logits, blank_idx=vocab.blank_idx)[0]
            text = vocab.decode(dec).strip()
            conf = float(probs.max(dim=2).values.mean().item())
        return text, round(conf, 3)
    except Exception as e:
        return "", 0.0

# Load references
tpl = json.load(open("annotations/templates/region_templates.json", encoding="utf-8"))
references = []
for src in tpl["manual_source_images"]:
    fn = src["filename"]
    p = Path("uploads") / fn
    if not p.exists():
        continue
    img = cv2.imread(str(p))
    kp, des, norm = image_features(img)
    references.append({
        "filename": fn,
        "kp": kp,
        "des": des,
        "norm": norm,
        "regions": src["regions"]
    })

print(f"Loaded {len(references)} template references with {sum(len(r['regions']) for r in references)} total region definitions.")

# Test on 2 sample uploads
sample_files = [
    "WhatsApp Image 2026-09-17 at 4.09.25 PM.jpeg",
    "WhatsApp Image 2026-09-17 at 4.09.23 PM (1).jpeg"
]

rotations = [(None, 0), (cv2.ROTATE_90_CLOCKWISE, 90), (cv2.ROTATE_180, 180), (cv2.ROTATE_90_COUNTERCLOCKWISE, 270)]

for sfn in sample_files:
    sp = Path("uploads") / sfn
    if not sp.exists():
        continue
    print(f"\n=======================================================")
    print(f"TESTING EXTRACTION ON: {sfn}")
    print(f"=======================================================")
    img = cv2.imread(str(sp))
    best = None
    best_inliers = 0
    for rot_code, angle in rotations:
        rot_img = img if rot_code is None else cv2.rotate(img, rot_code)
        kp, des, norm = image_features(rot_img)
        if des is None or len(kp) < 10:
            continue
        for ref in references:
            matcher = cv2.BFMatcher(norm)
            matches = matcher.knnMatch(des, ref["des"], k=2)
            good = [m for m, n in matches if len(matches) > 1 and m.distance < 0.75 * n.distance]
            if len(good) < 20:
                continue
            src_pts = np.float32([kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([ref["kp"][m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if H is None or mask is None:
                continue
            det = np.linalg.det(H[:2, :2])
            trace = H[0, 0] + H[1, 1]
            if det <= 0.1 or trace <= 0.4:
                continue
            inliers = int(mask.ravel().sum())
            if inliers > best_inliers:
                best_inliers = inliers
                best = {"angle": angle, "rot_img": rot_img, "ref": ref, "H": H, "inliers": inliers}
    
    if not best:
        print("FAILED TO ALIGN BILL!")
        continue
    
    rot_img = best["rot_img"]
    H_inv = np.linalg.inv(best["H"])
    img_h, img_w = rot_img.shape[:2]
    print(f"Aligned successfully! Angle: {best['angle']}° | Inliers: {best['inliers']} | Template: {best['ref']['filename']}")
    print(f"\nExtracted Fields from Image Pixels:")
    print(f"{'Type':<12} | {'Field':<25} | {'Conf':<6} | Extracted Text")
    print("-" * 75)
    
    extracted_count = 0
    for r in best["ref"]["regions"]:
        x, y, w, h = r["bbox_xywh"]
        ref_poly = np.float32([[x, y], [x + w, y], [x + w, y + h], [x, y + h]]).reshape(-1, 1, 2)
        mapped = cv2.perspectiveTransform(ref_poly, H_inv).reshape(-1, 2)
        min_x = max(0, int(mapped[:, 0].min()))
        min_y = max(0, int(mapped[:, 1].min()))
        max_x = min(img_w, int(mapped[:, 0].max()))
        max_y = min(img_h, int(mapped[:, 1].max()))
        cw, ch = max_x - min_x, max_y - min_y
        if cw < 10 or ch < 8:
            continue
        crop = rot_img[min_y:max_y, min_x:max_x]
        
        tt = r.get("text_type", "printed")
        if tt == "digits":
            text, conf = recognize_crop(crop, model_d, vocab_d)
        elif tt == "handwritten":
            text, conf = recognize_crop(crop, model_h, vocab_h)
        else:
            text, conf = recognize_crop(crop, model_p, vocab_p)
        
        if text:
            extracted_count += 1
            print(f"{tt:<12} | {r.get('region_key', ''):<25} | {conf:<6.2f} | {text}")
    print(f"\nTotal Extracted: {extracted_count} fields extracted dynamically from uploaded bill!")
