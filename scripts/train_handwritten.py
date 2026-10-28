#!/usr/bin/env python
"""
train_handwritten.py
Trains a dedicated CRNN model strictly for handwritten fields
(Farmer Name, Village, Mandal, Commodity, Commission Agent, Buyer Name).
Trained on 100% real crops from uploaded bill images.
No pretrained models. No synthetic data. No hardcoded predictions.
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


def load_hw_records():
    """Load all verified handwritten records from auto-cropped & manual annotations."""
    records = []
    source_dirs = [
        BASE_DIR / "annotations" / "reviewed" / "handwritten",
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
            if ann.get("type") != "handwritten":
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
    """Background-subtraction illumination normalization."""
    ksize = max(int(min(gray.shape) * 0.05) | 1, 11)
    dilated = cv2.dilate(gray, np.ones((5, 5), np.uint8))
    bg = cv2.medianBlur(dilated, ksize)
    bg_safe = np.maximum(bg.astype(np.float32), 1.0)
    return np.clip((gray.astype(np.float32) / bg_safe) * 255.0, 0, 255).astype(np.uint8)


def augment_image(gray: np.ndarray) -> np.ndarray:
    """Light brightness/contrast/noise/morphology augmentation."""
    if np.random.rand() < 0.4:
        delta = np.random.uniform(-15, 15)
        gray = np.clip(gray.astype(np.float32) + delta, 0, 255).astype(np.uint8)
    if np.random.rand() < 0.3:
        alpha = np.random.uniform(0.9, 1.1)
        gray = np.clip(gray.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
    if np.random.rand() < 0.2:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        if np.random.rand() < 0.5:
            gray = cv2.erode(gray, k, iterations=1)
        else:
            gray = cv2.dilate(gray, k, iterations=1)
    return gray


def prepare_crop(crop_path: str) -> np.ndarray:
    """Read crop, normalize, resize to canvas (32 x 320) with proportional scaling."""
    img = cv2.imread(crop_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {crop_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    norm = normalize_illumination(gray)
    h, w = norm.shape[:2]
    th = 32
    tw = 320  # standard canvas for handwritten Telugu words
    scale_w = max(24, min(int(w * th / float(max(h, 1))), tw))
    resized = cv2.resize(norm, (scale_w, th), interpolation=cv2.INTER_AREA)
    bg_val = float(np.median(resized))
    canvas = np.full((th, tw), bg_val, dtype=np.uint8)
    canvas[:, :scale_w] = resized
    return canvas


class HwCropDataset(Dataset):
    def __init__(self, records, vocab: OCRVocab, augment: bool = False):
        self.records = records
        self.vocab = vocab
        self.augment = augment
        self.cached_images = []
        self.cached_labels = []

        print(f"  [CACHE] Preloading {len(records)} handwritten crops into RAM...")
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
    flat_targets = torch.tensor([idx for l in labels for idx in l], dtype=torch.long)
    return imgs, flat_targets, in_lens, tgt_lens


def compute_metrics(model, dataloader, vocab: OCRVocab, device: torch.device):
    model.eval()
    total_cer = 0.0
    exact = 0
    n = 0
    with torch.no_grad():
        for imgs, targets, in_lens, tgt_lens in dataloader:
            imgs = imgs.to(device)
            logits = model(imgs)
            decoded = model.decode_greedy(logits, blank_idx=vocab.blank_idx)
            offset = 0
            for i, tlen in enumerate(tgt_lens.tolist()):
                gold_ids = targets[offset:offset + tlen].tolist()
                offset += tlen
                gold = vocab.decode(gold_ids)
                pred = vocab.decode(decoded[i]) if i < len(decoded) else ""
                ops = SequenceMatcher(None, gold, pred).get_opcodes()
                edits = sum(1 for op in ops if op[0] != "equal")
                total_cer += edits / max(len(gold), 1)
                if gold.strip() == pred.strip():
                    exact += 1
                n += 1
    return exact / max(n, 1), total_cer / max(n, 1)


def train(epochs: int = 45, batch_size: int = 16, lr: float = 0.002):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("CRNN Telugu Handwritten Model Training on Real Uploaded Bill Crops")
    print(f"Device: {device} | Epochs: {epochs} | Batch size: {batch_size} | LR: {lr}")
    print("=" * 70)

    records = load_hw_records()
    if not records:
        print("[ERROR] No handwritten records found.")
        return

    char_counts = Counter(ch for r in records for ch in r["text"])
    hw_vocab_chars = sorted(char_counts.keys())
    vocab = OCRVocab(characters=hw_vocab_chars)
    print(f"Loaded {len(records)} real crops.")
    print(f"Pruned Handwritten Vocabulary ({len(vocab)} classes): {hw_vocab_chars}")

    np.random.seed(42)
    indices = np.random.permutation(len(records))
    split_idx = int(len(records) * 0.85)
    train_records = [records[i] for i in indices[:split_idx]]
    val_records = [records[i] for i in indices[split_idx:]]

    train_ds = HwCropDataset(train_records, vocab, augment=True)
    val_ds = HwCropDataset(val_records, vocab, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model = CRNN(
        img_h=32,
        in_channels=1,
        num_classes=len(vocab),
        hidden_size=128,
        num_lstm_layers=2,
        dropout=0.1
    ).to(device)

    with torch.no_grad():
        model.fc.bias.data[vocab.blank_idx] = -3.0

    ctc_loss = nn.CTCLoss(blank=vocab.blank_idx, zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    total_steps = epochs * len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=total_steps, pct_start=0.15)

    best_score = -1.0
    best_val_cer = 999.0
    best_exact_match = 0.0
    best_state = None

    best_checkpoint_path = SAVED_MODELS_DIR / "ocr_handwritten.pth"
    vocab_save_path = SAVED_MODELS_DIR / "ocr_handwritten_vocab.json"
    vocab.save(str(vocab_save_path))

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for imgs, targets, in_lens, tgt_lens in train_loader:
            imgs = imgs.to(device)
            targets = targets.to(device)
            in_lens = in_lens.to(device)
            tgt_lens = tgt_lens.to(device)

            optimizer.zero_grad()
            logits = model(imgs)  # (T, B, C)
            log_probs = logits.log_softmax(2)
            loss = ctc_loss(log_probs, targets, in_lens, tgt_lens)
            if torch.isfinite(loss) and loss.item() > 0:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()

        avg_loss = total_loss / max(1, len(train_loader))

        if epoch % 5 == 0 or epoch == epochs:
            val_acc, val_cer = compute_metrics(model, val_loader, vocab, device)
            score = (1.0 - val_cer) + 2.0 * val_acc
            tag = ""
            if score > best_score:
                best_score = score
                best_val_cer = val_cer
                best_exact_match = val_acc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                tag = " [BEST]"
                torch.save({
                    "model_state_dict": best_state,
                    "epoch": epoch,
                    "training_source": "manual_user_images",
                    "validation_cer": val_cer,
                    "exact_match": val_acc,
                    "num_classes": len(vocab),
                    "created_at": datetime.now(timezone.utc).isoformat()
                }, best_checkpoint_path)

            print(f"Epoch {epoch:02d}/{epochs:02d} | Loss: {avg_loss:.4f} | Val Exact: {val_acc*100:.1f}% | Val CER: {val_cer:.4f}{tag}")
            sys.stdout.flush()

    print("=" * 70)
    print(f"Handwritten Telugu CRNN Training Finished!")
    print(f"Best Val Exact Match: {best_exact_match*100:.1f}% | Best Val CER: {best_val_cer:.4f}")
    print(f"Saved Checkpoint: {best_checkpoint_path}")
    print(f"Saved Vocabulary: {vocab_save_path}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.002)
    args = parser.parse_args()
    train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
