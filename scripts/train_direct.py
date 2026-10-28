"""
train_direct.py -- Direct CRNN training from all available annotations.

Data sources used (in priority order):
  1. data_generation/ocr_annotations.jsonl (manually transcribed, highest quality)
  2. annotations/automatic/*/records/*.json (auto-generated from alignment, with
     region_type labels -- used only when the crop file exists on disk)

No synthetic data. No pretrained weights. No external APIs.

Usage:
    python scripts/train_direct.py
    python scripts/train_direct.py --epochs 200 --lr 0.0005
"""
import argparse
import json
import os
import random
import sys
import tempfile
from pathlib import Path
from datetime import datetime, timezone

# ── Fix Windows console encoding so Telugu/Unicode chars print without crash ──
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))


import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import config
from models.ocr_crnn import CRNN
from models.vocab import OCRVocab
from preprocessing.ocr_transform import prepare_ocr_crop, split_vertical_crop_into_lines


# ─────────────────────────────────────────────────────────────
#  Data loading
# ─────────────────────────────────────────────────────────────

def load_manual_annotations(ann_path: str) -> list:
    """
    Source 1: Load manually transcribed Telugu/English text from
    data_generation/ocr_annotations.jsonl.
    These have real, human-typed ground-truth text.
    """
    records = []
    path = Path(ann_path)
    if not path.exists():
        print(f"[WARN] Manual annotation file not found: {ann_path}")
        return records

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue

            text = r.get("text", "").strip()
            crop_path = r.get("crop_path", "")
            if not text or not crop_path:
                continue
            # Skip pure separator records
            if set(text) <= set("-_ ."):
                continue

            abs_crop = Path(crop_path) if os.path.isabs(crop_path) else BASE_DIR / crop_path
            if not abs_crop.exists():
                continue

            bill_group = r.get("source_filename", r.get("source_image", str(abs_crop.parent)))
            records.append({
                "id": r.get("id", ""),
                "text": text,
                "crop_path": str(abs_crop),
                "bill_group": bill_group,
                "source": "manual",
                "priority": 1,  # highest priority, always use in train
            })

    print(f"[INFO] Manual annotations: {len(records)} usable records")
    return records


def expand_vertical_crops(records: list, save_dir: str) -> list:
    """
    Key fix for accuracy: annotation crops from sideways bill images are tall-and-narrow
    vertical strips containing MULTIPLE text lines stacked. Before training, we must:
      1. Rotate each vertical crop 90 deg CCW to make it horizontal
      2. Segment the horizontal strip into individual text lines
      3. Associate the same text label with EACH line (or keep only viable ones)

    This converts e.g. a 748x45 vertical crop (ratio=16) labeled
    'రైతు/అమ్ముకందారు పేరు.' into N horizontal line sub-crops.

    For crops that ARE already horizontal (ratio <= 1.15), they pass through unchanged.
    """
    import tempfile, hashlib
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    expanded = []
    n_split = 0
    n_passthrough = 0

    for rec in records:
        crop_path = rec.get("crop_path", "")
        if not crop_path or not Path(crop_path).exists():
            expanded.append(rec)
            continue

        img = cv2.imread(crop_path, cv2.IMREAD_COLOR)
        if img is None:
            expanded.append(rec)
            continue

        h, w = img.shape[:2]
        ratio = h / max(w, 1)

        # Only split genuinely vertical crops
        if ratio <= 1.15:
            expanded.append(rec)
            n_passthrough += 1
            continue

        # Split the vertical crop into horizontal text lines
        line_crops = split_vertical_crop_into_lines(img)

        if not line_crops or len(line_crops) <= 1:
            # Could not segment; use rotated version as-is
            rotated = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            key = hashlib.md5(f"{crop_path}_rotated".encode()).hexdigest()[:16]
            out_path = save_path / f"split_{key}.png"
            if not out_path.exists():
                cv2.imwrite(str(out_path), rotated)
            new_rec = dict(rec)
            new_rec["crop_path"] = str(out_path)
            expanded.append(new_rec)
            n_split += 1
            continue

        # Create one training record per line crop
        for line_idx, line_img in enumerate(line_crops):
            lh, lw = line_img.shape[:2]
            if lh < 8 or lw < 12 or line_img.size == 0:
                continue
            key = hashlib.md5(f"{crop_path}_{line_idx}".encode()).hexdigest()[:16]
            out_path = save_path / f"split_{key}.png"
            if not out_path.exists():
                cv2.imwrite(str(out_path), line_img)
            new_rec = dict(rec)
            new_rec["crop_path"] = str(out_path)
            expanded.append(new_rec)
        n_split += 1

    print(
        f"[INFO] Crop expansion: {n_passthrough} horizontal (unchanged), "
        f"{n_split} vertical crops split into {len(expanded) - n_passthrough} records"
    )
    return expanded


def load_automatic_annotations(auto_root: str, manual_annotations: list) -> list:
    """
    Source 2: Load auto-generated alignment crops that have text labels.
    Map source_annotation_ids to actual ground truth Telugu/English text from manual annotations.
    Do NOT fall back to English region names when the crop contains Telugu text.
    """
    records = []
    auto_root_path = Path(auto_root)
    if not auto_root_path.exists():
        return records

    # Build manual id -> text mapping
    text_lookup = {}
    for m in manual_annotations:
        m_id = m.get("id") or Path(m.get("crop_path", "")).stem
        if m.get("text"):
            text_lookup[m_id] = m["text"]
            if m.get("id"):
                text_lookup[m["id"]] = m["text"]

    # Also load from reviewed manual annotations directory
    reviewed_manual_dir = BASE_DIR / "annotations" / "reviewed" / "manual"
    if reviewed_manual_dir.exists():
        for mf in reviewed_manual_dir.glob("*.json"):
            try:
                md = json.loads(mf.read_text(encoding="utf-8"))
                if md.get("id") and md.get("text"):
                    text_lookup[md["id"]] = md["text"]
            except Exception:
                pass

    json_files = sorted(auto_root_path.glob("*/records/*.json"))
    print(f"[INFO] Scanning {len(json_files)} automatic annotation record files...")

    seen_crops = set()  # deduplicate by crop path

    for jf in json_files:
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            continue

        source_image = data.get("source_image", "")
        bill_group = Path(source_image).stem if source_image else jf.stem

        # Extract regions that have valid ground truth text
        for r in data.get("regions", []):
            cp = r.get("crop_path", "")
            if not cp:
                continue
            abs_cp = Path(cp) if os.path.isabs(cp) else BASE_DIR / cp
            if not abs_cp.exists() or str(abs_cp) in seen_crops:
                continue

            text = r.get("text", "").strip()
            if not text:
                ann_ids = r.get("source_annotation_ids", [])
                for aid in ann_ids:
                    if aid in text_lookup and text_lookup[aid].strip():
                        text = text_lookup[aid].strip()
                        break

            # Only use if legitimate ground truth text exists
            if not text or set(text) <= set("-_ ."):
                continue

            seen_crops.add(str(abs_cp))
            records.append({
                "text": text,
                "crop_path": str(abs_cp),
                "bill_group": bill_group,
                "source": "automatic_aligned",
                "priority": 2,
            })

    print(f"[INFO] Automatic annotations: {len(records)} usable records with ground truth text")
    return records


def combine_and_deduplicate(manual: list, automatic: list) -> list:
    """
    Merge manual + automatic, keeping manual records authoritative.
    Deduplicate by crop_path (manual wins over automatic for same path).
    """
    seen = {}
    for r in manual:
        seen[r["crop_path"]] = r
    for r in automatic:
        if r["crop_path"] not in seen:
            seen[r["crop_path"]] = r
    combined = list(seen.values())
    manual_count = sum(1 for r in combined if r["source"] == "manual")
    auto_count = len(combined) - manual_count
    print(f"[INFO] Combined dataset: {len(combined)} total crops "
          f"({manual_count} manual, {auto_count} automatic)")
    return combined


def split_by_bill_group(records: list, val_frac: float = 0.15, seed: int = 42):
    """
    Bill-level split to prevent data leakage.
    Manual-source records always go to train (they are too few to hold out).
    Automatic records are split: ~85% train, ~15% val.
    """
    # Always include ALL manual records in training
    manual_recs = [r for r in records if r["source"] == "manual"]
    auto_recs = [r for r in records if r["source"] != "manual"]

    # Split automatic records by bill group
    auto_groups = sorted({r["bill_group"] for r in auto_recs})
    rng = random.Random(seed)
    rng.shuffle(auto_groups)

    n_val = max(1, round(len(auto_groups) * val_frac)) if len(auto_groups) > 1 else 0
    val_groups = set(auto_groups[:n_val])

    train = manual_recs + [r for r in auto_recs if r["bill_group"] not in val_groups]
    val = [r for r in auto_recs if r["bill_group"] in val_groups]

    # If no val set, use 15% of train as val (without removing from train,
    # since with this small dataset we need every sample for training too)
    if not val:
        rng2 = random.Random(seed + 1)
        val = rng2.sample(train, max(1, len(train) // 6))

    print(f"[INFO] Train: {len(train)} crops | Val: {len(val)} crops")
    return train, val


# ─────────────────────────────────────────────────────────────
#  Dataset with augmentation
# ─────────────────────────────────────────────────────────────

class DirectOCRDataset(Dataset):
    def __init__(self, records: list, vocab: OCRVocab, augment: bool = False):
        self.records = records
        self.vocab = vocab
        self.augment = augment

        # Pre-load all images into RAM to eliminate disk I/O bottleneck.
        # With ~1,700 crops averaging ~50KB each, total RAM use is ~85MB — acceptable.
        print(f"[INFO] Pre-loading {len(records)} images into RAM cache...", flush=True)
        self.image_cache = []
        missing = 0
        for rec in records:
            img = cv2.imread(rec["crop_path"], cv2.IMREAD_COLOR)
            if img is None or img.size == 0:
                img = np.full((32, 64, 3), 200, dtype=np.uint8)
                missing += 1
            self.image_cache.append(img)
        if missing:
            print(f"[WARN] {missing} images could not be read; using blank placeholders.", flush=True)
        print(f"[INFO] Image cache ready. Starting training...", flush=True)

    def __len__(self):
        return len(self.records)

    def _augment(self, img: np.ndarray) -> np.ndarray:
        # Brightness / contrast jitter
        alpha = random.uniform(0.70, 1.40)
        beta = random.randint(-25, 25)
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        # Gaussian noise
        if random.random() < 0.45:
            noise = np.random.normal(0, 9, img.shape).astype(np.float32)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        # Slight blur
        if random.random() < 0.35:
            k = random.choice([3, 5])
            img = cv2.GaussianBlur(img, (k, k), 0)
        # Horizontal scaling jitter (simulate camera distance variation)
        if random.random() < 0.3:
            h, w = img.shape[:2]
            scale = random.uniform(0.85, 1.15)
            new_w = max(16, int(w * scale))
            img = cv2.resize(img, (new_w, h), interpolation=cv2.INTER_LINEAR)
        return img

    def __getitem__(self, idx):
        rec = self.records[idx]
        # Use cached image (already in RAM — no disk I/O)
        img = self.image_cache[idx].copy()
        if self.augment:
            img = self._augment(img)
        tensor = prepare_ocr_crop(img)
        target = torch.tensor(self.vocab.encode(rec["text"]), dtype=torch.long)
        # Sequence length = OCR_IMG_W / 4  (two (2,1) MaxPool layers halve width twice)
        input_length = config.OCR_IMG_W // 4
        return tensor, target, input_length, len(target)



def collate_fn(batch):
    images, targets, input_lengths, target_lengths = zip(*batch)
    images = torch.stack(images, 0)
    input_lengths = torch.tensor(input_lengths, dtype=torch.long)
    target_lengths = torch.tensor(target_lengths, dtype=torch.long)
    max_tl = max(t.size(0) for t in targets)
    padded = torch.zeros(len(targets), max_tl, dtype=torch.long)
    for i, t in enumerate(targets):
        padded[i, : t.size(0)] = t
    return images, padded, input_lengths, target_lengths


# ─────────────────────────────────────────────────────────────
#  Metrics
# ─────────────────────────────────────────────────────────────

def levenshtein(ref: str, hyp: str) -> int:
    prev = list(range(len(hyp) + 1))
    for rc in ref:
        cur = [prev[0] + 1]
        for j, hc in enumerate(hyp, 1):
            cur.append(min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + (rc != hc)))
        prev = cur
    return prev[-1]


def evaluate(model, loader, vocab, device):
    model.eval()
    char_err = char_tot = exact = total = 0
    with torch.no_grad():
        for imgs, targets, in_lens, tgt_lens in loader:
            logits = model(imgs.to(device))
            preds = model.decode_greedy(logits, blank_idx=vocab.blank_idx)
            for b in range(len(preds)):
                hyp = vocab.decode(preds[b]).strip()
                ref = vocab.decode(targets[b][:tgt_lens[b]].tolist()).strip()
                char_err += levenshtein(ref, hyp)
                char_tot += max(len(ref), 1)
                exact += int(ref == hyp)
                total += 1
    model.train()
    return {
        "cer": round(char_err / max(char_tot, 1), 4),
        "exact": round(exact / max(total, 1), 4),
        "samples": total,
    }


# ─────────────────────────────────────────────────────────────
#  Checkpoint saver
# ─────────────────────────────────────────────────────────────

def safe_save(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".pth", delete=False) as h:
            tmp = h.name
        torch.save(data, tmp)
        os.replace(tmp, path)
    finally:
        if tmp and os.path.exists(tmp):
            os.remove(tmp)


# ─────────────────────────────────────────────────────────────
#  Main training loop
# ─────────────────────────────────────────────────────────────

def train(epochs: int, batch_size: int, lr: float, patience: int, seed: int):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # 1. Load all available real data
    manual = load_manual_annotations(config.OCR_ANNOTATIONS_PATH)
    automatic = load_automatic_annotations(
        str(BASE_DIR / "annotations" / "automatic"),
        manual_annotations=manual,
    )
    all_records = combine_and_deduplicate(manual, automatic)

    # 1b. Expand vertical multi-line crops into per-line training samples
    # This is the KEY fix: annotation crops are tall vertical strips from
    # rotated bill images. Each contains multiple stacked text lines.
    # We split them into individual horizontal lines before training.
    split_cache_dir = str(BASE_DIR / "data_generation" / "split_line_crops")
    all_records = expand_vertical_crops(all_records, save_dir=split_cache_dir)

    if not all_records:
        print("[ERROR] No annotation records found at all. Check your data.")
        return

    # 2. Split into train/val
    train_recs, val_recs = split_by_bill_group(all_records, val_frac=0.15, seed=seed)

    # 3. Use complete multilingual vocabulary (177 classes: Telugu vowels, consonants, matras, digits 0-9, English, symbols)
    vocab = OCRVocab()
    all_chars = [vocab.idx2char[i] for i in range(len(vocab))]
    print(f"[INFO] Vocabulary: {len(vocab)} classes (full Telugu, digits 0-9, English, symbols)")

    # Verify OOV characters
    for r in all_records:
        for ch in r["text"]:
            if not vocab.contains(ch):
                print(f"  [WARN] OOV: U+{ord(ch):04X} '{ch}' in {r['text']!r}")

    # 4. Datasets and loaders
    train_ds = DirectOCRDataset(train_recs, vocab, augment=True)
    val_ds = DirectOCRDataset(val_recs, vocab, augment=False)
    eff_batch = min(batch_size, len(train_recs))

    # Use multiple workers to parallelize PNG disk reads (main bottleneck on CPU)
    # Windows requires num_workers >= 0; set to 0 if multiprocessing causes issues
    n_workers = 0  # Windows-safe: avoids multiprocessing pickle issues with OpenCV
    train_loader = DataLoader(
        train_ds, batch_size=eff_batch, shuffle=True,
        collate_fn=collate_fn, drop_last=False,
        num_workers=n_workers,
        prefetch_factor=None if n_workers == 0 else 2,
    )
    val_loader = DataLoader(
        val_ds, batch_size=min(batch_size, max(len(val_recs), 1)),
        shuffle=False, collate_fn=collate_fn,
        num_workers=n_workers,
        prefetch_factor=None if n_workers == 0 else 2,
    )

    # 5. Model
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
        pct_start=0.12, anneal_strategy="cos"
    )
    criterion = nn.CTCLoss(blank=vocab.blank_idx, zero_infinity=True)

    model_dir = Path(config.SAVED_MODELS_DIR)
    best_path = model_dir / "ocr_crnn.pth"
    vocab_path = Path(config.OCR_VOCAB_PATH)

    print(f"\n{'='*65}")
    print(f"Training CRNN OCR | {len(train_recs)} train, {len(val_recs)} val crops")
    print(f"Device: {device} | Epochs: {epochs} | LR: {lr} | Batch: {eff_batch}")
    print(f"{'='*65}\n")

    best_cer = float("inf")
    stale = 0
    best_state = None

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
            if cer < best_cer:
                best_cer = cer
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
                stale = 0
                tag = "BEST"
            else:
                stale += 1
                tag = f"stale {stale}/{patience}"
            print(f"Epoch {epoch:4d}/{epochs}  loss={avg_loss:.4f}  "
                  f"val_CER={cer:.4f}  exact={acc:.3f}  [{tag}]")

            if stale >= patience and epoch > 20:
                print(f"[INFO] Early stopping at epoch {epoch}")
                break

    # Save best checkpoint
    if best_state is not None:
        checkpoint = {
            "model_state_dict": best_state,
            "training_source": "manual_user_images",   # Required by OCREngine
            "num_classes": len(vocab),
            "hidden_size": config.OCR_HIDDEN_SIZE,
            "num_layers": config.OCR_NUM_LAYERS,
            "validation_cer": best_cer,
            "training_images": len({r["bill_group"] for r in train_recs}),
            "training_crops": len(train_recs),
            "all_chars": all_chars,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        safe_save(best_path, checkpoint)
        vocab.save(str(vocab_path))
        pct = (1.0 - best_cer) * 100
        print(f"\n[OK] Saved model -> {best_path}")
        print(f"[OK] Saved vocab -> {vocab_path}")
        print(f"[OK] Best val CER={best_cer:.4f} => ~{pct:.1f}% char accuracy")
        print(f"[OK] Training crops: {len(train_recs)} | Val crops: {len(val_recs)}")
    else:
        print("[ERROR] Training did not produce a valid checkpoint.")


def main():
    parser = argparse.ArgumentParser(
        description="Train CRNN OCR from all available real annotated data"
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args.epochs, args.batch_size, args.lr, args.patience, args.seed)


if __name__ == "__main__":
    main()
