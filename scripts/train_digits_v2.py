#!/usr/bin/env python
"""
train_digits_v2.py
Trains CRNN strictly on user uploaded bill digit crops.
Architecture & pipeline 100% mirrors the proven train_printed.py pipeline:
  - Canvas: 32 x 320 (left aligned)
  - CRNN: 7-layer CNN + 2-layer BiLSTM (hidden_size=192)
  - AdamW + OneCycleLR
  - 100% real crops from uploaded bill images (zero synthetic, zero pretrained)
"""
import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

sys.stdout.reconfigure(encoding='utf-8')
import config
from models.ocr_crnn import CRNN
from models.vocab import OCRVocab

SAVED_MODELS_DIR = Path(config.SAVED_MODELS_DIR)
SAVED_MODELS_DIR.mkdir(exist_ok=True)


def load_digit_records():
    """Load all verified digit records from auto-cropped & manual annotations."""
    records = []
    source_dirs = [
        BASE_DIR / "annotations" / "reviewed" / "digits",
        BASE_DIR / "annotations" / "reviewed" / "manual"
    ]
    seen_crops = set()
    for sdir in source_dirs:
        if not sdir.exists():
            continue
        for jf in sdir.glob("*.json"):
            try:
                ann = json.loads(jf.read_text(encoding="utf-8"))
            except Exception:
                continue
            if ann.get("type") != "digits":
                continue
            text = (ann.get("text") or "").strip()
            if not text:
                continue
            crop_path = ann.get("crop_path", "")
            abs_cp = BASE_DIR / crop_path if not os.path.isabs(crop_path) else Path(crop_path)
            if not abs_cp.exists() or str(abs_cp) in seen_crops:
                continue
            seen_crops.add(str(abs_cp))
            records.append({
                "text": text,
                "crop_path": str(abs_cp),
                "field_name": ann.get("field_name", ""),
                "source": sdir.name
            })
    return records


def normalize_illumination(gray: np.ndarray) -> np.ndarray:
    ksize = max(int(min(gray.shape) * 0.05) | 1, 11)
    dilated = cv2.dilate(gray, np.ones((5, 5), np.uint8))
    bg = cv2.medianBlur(dilated, ksize)
    bg_safe = np.maximum(bg.astype(np.float32), 1.0)
    return np.clip((gray.astype(np.float32) / bg_safe) * 255.0, 0, 255).astype(np.uint8)


def augment_image(gray: np.ndarray) -> np.ndarray:
    if np.random.rand() < 0.4:
        delta = np.random.uniform(-20, 20)
        gray = np.clip(gray.astype(np.float32) + delta, 0, 255).astype(np.uint8)
    if np.random.rand() < 0.3:
        alpha = np.random.uniform(0.85, 1.15)
        gray = np.clip(gray.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
    if np.random.rand() < 0.25:
        noise = np.random.normal(0, 3, gray.shape).astype(np.float32)
        gray = np.clip(gray.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if np.random.rand() < 0.2:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        if np.random.rand() < 0.5:
            gray = cv2.erode(gray, k, iterations=1)
        else:
            gray = cv2.dilate(gray, k, iterations=1)
    return gray


def char_edits(s1: str, s2: str) -> int:
    """Standard Levenshtein distance for character error count."""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[m][n]


def prepare_crop(crop_path: str) -> np.ndarray:
    img = cv2.imread(crop_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {crop_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    if gray.shape[0] > int(gray.shape[1] * 1.15):
        gray = cv2.rotate(gray, cv2.ROTATE_90_COUNTERCLOCKWISE)
    norm = normalize_illumination(gray)
    th = config.OCR_IMG_H  # 32
    tw = 128               # Dense 128px canvas: eliminates dead padding for digits
    resized = cv2.resize(norm, (tw, th), interpolation=cv2.INTER_AREA)
    return resized



class DigitsCropDataset(Dataset):
    def __init__(self, records, vocab: OCRVocab, augment: bool = False):
        self.records = records
        self.vocab = vocab
        self.augment = augment
        self.cached_images = []
        self.cached_labels = []

        print(f"  [CACHE] Preloading {len(records)} digit crops into RAM...")
        for rec in records:
            canvas = prepare_crop(rec["crop_path"])
            label = self.vocab.encode(rec["text"])
            self.cached_images.append(canvas)
            self.cached_labels.append(label)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        canvas = self.cached_images[idx]
        if self.augment:
            canvas = augment_image(canvas)
        normalized = (canvas.astype(np.float32) / 127.5) - 1.0
        tensor = torch.from_numpy(normalized).unsqueeze(0).float()
        label = self.cached_labels[idx]
        return tensor, label


def collate_fn(batch):
    imgs, labels = zip(*batch)
    imgs = torch.stack(imgs, 0)
    in_lens = torch.full((len(imgs),), imgs.shape[3] // 4, dtype=torch.long)
    tgt_lens = torch.tensor([len(l) for l in labels], dtype=torch.long)
    flat = [idx for label in labels for idx in label]
    targets = torch.tensor(flat, dtype=torch.long)
    return imgs, targets, in_lens, tgt_lens


def evaluate(model, loader, vocab, device):
    model.eval()
    total_cer = 0.0
    exact = 0
    n = 0
    with torch.no_grad():
        for imgs, targets, in_lens, tgt_lens in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            decoded = model.decode_greedy(logits, blank_idx=vocab.blank_idx)
            offset = 0
            for i, tlen in enumerate(tgt_lens.tolist()):
                gold_ids = targets[offset:offset + tlen].tolist()
                offset += tlen
                gold = vocab.decode(gold_ids).strip()
                pred = vocab.decode(decoded[i]).strip() if i < len(decoded) else ""
                edits = char_edits(gold, pred)
                total_cer += edits / max(len(gold), 1)
                if gold == pred and len(gold) > 0:
                    exact += 1
                n += 1
    return {"cer": total_cer / max(n, 1), "exact": exact / max(n, 1)}



def train(epochs: int = 40, batch_size: int = 16, lr: float = 0.001):
    print("[INFO] Loading digits annotation records...")
    records = load_digit_records()
    if not records:
        print("[ERROR] No digit crops found.")
        return

    dataset_chars = sorted(list(set(c for r in records for c in r["text"])))
    vocab = OCRVocab(characters=dataset_chars)
    print(f"[INFO] Loaded {len(records)} digit crops | Vocab: {len(vocab)} classes: {dataset_chars}")

    np.random.seed(42)
    indices = np.random.permutation(len(records))
    val_n = max(1, len(records) // 5)
    train_recs = [records[i] for i in indices[val_n:]]
    val_recs = [records[i] for i in indices[:val_n]]
    print(f"[INFO] Train: {len(train_recs)} | Val: {len(val_recs)}")

    train_ds = DigitsCropDataset(train_recs, vocab, augment=True)
    val_ds = DigitsCropDataset(val_recs, vocab, augment=False)

    eff_batch = min(batch_size, len(train_recs))
    train_loader = DataLoader(train_ds, batch_size=eff_batch, shuffle=True,
                              collate_fn=collate_fn, drop_last=False, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=min(batch_size, max(len(val_recs), 1)),
                            shuffle=False, collate_fn=collate_fn, num_workers=0)

    device = config.DEVICE
    model = CRNN(
        img_h=config.OCR_IMG_H,
        in_channels=config.OCR_IN_CHANNELS,
        num_classes=len(vocab),
        hidden_size=config.OCR_HIDDEN_SIZE,
        num_lstm_layers=config.OCR_NUM_LAYERS,
        dropout=config.OCR_DROPOUT,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    total_steps = epochs * max(len(train_loader), 1)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=total_steps,
        pct_start=0.15, anneal_strategy="cos"
    )
    criterion = nn.CTCLoss(blank=vocab.blank_idx, zero_infinity=True)

    print(f"\nTraining CRNN Digits | {len(train_recs)} train, {len(val_recs)} val | {epochs} epochs")
    sys.stdout.flush()

    best_cer = float("inf")
    best_acc = 0.0
    best_score = -1.0
    best_state = None
    all_chars_list = [vocab.idx2char[i] for i in range(len(vocab))]

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for imgs, targets, in_lens, tgt_lens in train_loader:
            imgs = imgs.to(device)
            targets = targets.to(device)
            in_lens = in_lens.to(device)
            tgt_lens = tgt_lens.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits.log_softmax(2), targets, in_lens, tgt_lens)
            if torch.isfinite(loss) and loss.item() > 0:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
                n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)

        if epoch % 5 == 0 or epoch <= 5 or epoch == epochs:
            m = evaluate(model, val_loader, vocab, device)
            cer, acc = m["cer"], m["exact"]
            score = (1.0 - min(cer, 1.0)) * 0.5 + acc * 2.0
            tag = "stale"
            if score > best_score:
                best_score = score
                best_cer = cer
                best_acc = acc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                tag = "BEST"

            print(f"Epoch {epoch:4d}/{epochs}  loss={avg_loss:.4f}  val_CER={cer:.4f}  exact={acc:.3f}  [{tag}]")
            sys.stdout.flush()

    if best_state is not None:
        model_path = SAVED_MODELS_DIR / "ocr_digits.pth"
        vocab_path = SAVED_MODELS_DIR / "ocr_digits_vocab.json"
        checkpoint = {
            "model_state_dict": best_state,
            "training_source": "manual_user_images",
            "num_classes": len(vocab),
            "hidden_size": config.OCR_HIDDEN_SIZE,
            "num_layers": config.OCR_NUM_LAYERS,
            "validation_cer": best_cer,
            "exact_accuracy": best_acc,
            "training_crops": len(train_recs),
            "all_chars": all_chars_list,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        torch.save(checkpoint, str(model_path))
        vocab.save(str(vocab_path))
        pct = (1.0 - best_cer) * 100
        print(f"\n[OK] Model saved -> {model_path}")
        print(f"[OK] Vocab saved  -> {vocab_path}")
        print(f"[OK] Best val CER={best_cer:.4f} => ~{pct:.1f}% char accuracy | exact={best_acc:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()
    train(args.epochs, args.batch_size, args.lr)
