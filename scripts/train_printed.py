#!/usr/bin/env python
"""
train_printed.py
Trains CRNN on ONLY printed-type annotation crops from user manual annotations.
No handwritten labels. No label copying across bills.
Run: python scripts/train_printed.py --epochs 40
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding='utf-8')
import config
from models.ocr_crnn import CRNN
from models.vocab import OCRVocab

BASE_DIR = Path(config.BASE_DIR)
ANNOTATIONS_DIR = BASE_DIR / "annotations" / "reviewed" / "manual"
SAVED_MODELS_DIR = Path(config.SAVED_MODELS_DIR)
SAVED_MODELS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_printed_records():
    """Load printed-type annotations from both manual annotations and auto-transferred annotations."""
    records = []
    source_dirs = [
        BASE_DIR / "annotations" / "reviewed" / "manual",
        BASE_DIR / "annotations" / "reviewed" / "auto"
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
            if ann.get("type") != "printed":
                continue
            text = ann.get("text", "").strip()
            if not text or set(text) <= set("-_ ."):
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


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------
def normalize_illumination(gray: np.ndarray) -> np.ndarray:
    """Background-subtraction illumination normalization."""
    ksize = max(int(min(gray.shape) * 0.05) | 1, 11)
    dilated = cv2.dilate(gray, np.ones((5, 5), np.uint8))
    bg = cv2.medianBlur(dilated, ksize)
    bg_safe = np.maximum(bg.astype(np.float32), 1.0)
    return np.clip((gray.astype(np.float32) / bg_safe) * 255.0, 0, 255).astype(np.uint8)


def augment_image(gray: np.ndarray) -> np.ndarray:
    """Light brightness/contrast/noise/morphology augmentation for training robustness."""
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


def prepare_crop(crop_path: str) -> np.ndarray:
    """Read crop, normalize, resize to (OCR_IMG_H, OCR_IMG_W) canvas."""
    img = cv2.imread(crop_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {crop_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    norm = normalize_illumination(gray)
    h, w = norm.shape[:2]
    th = config.OCR_IMG_H
    tw = config.OCR_IMG_W
    scale_w = max(16, min(int(w * th / float(max(h, 1))), tw))
    resized = cv2.resize(norm, (scale_w, th), interpolation=cv2.INTER_AREA)
    bg_val = float(np.median(resized))
    canvas = np.full((th, tw), bg_val, dtype=np.uint8)
    canvas[:, :scale_w] = resized
    return canvas


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class PrintedCropDataset(Dataset):
    def __init__(self, records, vocab: OCRVocab, augment: bool = False):
        self.records = records
        self.vocab = vocab
        self.augment = augment
        self.cached_images = []
        self.cached_labels = []

        print(f"  [CACHE] Preloading {len(records)} crops into RAM...")
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
    # CTC input lengths = sequence length after CNN (width // 4 approx)
    in_lens = torch.full((len(imgs),), imgs.shape[3] // 4, dtype=torch.long)
    tgt_lens = torch.tensor([len(l) for l in labels], dtype=torch.long)
    flat = [idx for label in labels for idx in label]
    targets = torch.tensor(flat, dtype=torch.long)
    return imgs, targets, in_lens, tgt_lens


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
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
                gold = vocab.decode(gold_ids)
                pred = vocab.decode(decoded[i]) if i < len(decoded) else ""
                ops = SequenceMatcher(None, gold, pred).get_opcodes()
                edits = sum(1 for op in ops if op[0] != "equal")
                total_cer += edits / max(len(gold), 1)
                if gold == pred:
                    exact += 1
                n += 1
    return {"cer": total_cer / max(n, 1), "exact": exact / max(n, 1)}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(epochs: int, batch_size: int, lr: float):
    print("[INFO] Loading printed annotation records...")
    records = load_printed_records()
    if not records:
        print("[ERROR] No printed crops found. Run: python scripts/regen_crops.py first.")
        return

    sources = Counter(r.get("source", "unknown") for r in records)
    print(f"[INFO] Loaded {len(records)} total printed crops (Sources: {dict(sources)})")
    print(f"[INFO] Sample fields:")
    for r in records[:5]:
        print(f"  [{r.get('source')}] {r['field_name']:25s}  {repr(r['text'])}")

    dataset_chars = sorted(list(set(c for r in records for c in r["text"])))
    vocab = OCRVocab(characters=dataset_chars)
    print(f"[INFO] Focused Vocabulary: {len(vocab)} classes (from {len(dataset_chars)} unique characters)")

    # 80/20 train/val split
    np.random.seed(42)
    indices = np.random.permutation(len(records))
    val_n = max(1, len(records) // 5)
    train_recs = [records[i] for i in indices[val_n:]]
    val_recs = [records[i] for i in indices[:val_n]]
    print(f"[INFO] Train: {len(train_recs)} | Val: {len(val_recs)}")

    train_ds = PrintedCropDataset(train_recs, vocab, augment=True)
    val_ds = PrintedCropDataset(val_recs, vocab, augment=False)

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

    print(f"\n{'='*60}")
    print(f"Training CRNN | {len(train_recs)} train, {len(val_recs)} val | {epochs} epochs")
    print(f"Device: {device} | LR: {lr} | Batch: {eff_batch}")
    print(f"{'='*60}\n")
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
            # Combined score: higher exact match and lower CER
            score = (1.0 - cer) + 2.0 * acc
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
        model_path = SAVED_MODELS_DIR / "ocr_crnn.pth"
        vocab_path = Path(config.OCR_VOCAB_PATH)
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
    else:
        print("[ERROR] Training did not produce a valid checkpoint.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train CRNN on printed annotation crops only")
    parser.add_argument("--epochs", type=int, default=40,
                        help="Number of training epochs (35-50 recommended)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()
    train(args.epochs, args.batch_size, args.lr)


if __name__ == "__main__":
    main()
