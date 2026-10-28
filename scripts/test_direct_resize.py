import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import cv2
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(".").resolve()))
from models.ocr_crnn import CRNN
from models.vocab import OCRVocab
from scripts.train_digits import load_digit_records, normalize_illumination

records = load_digit_records()
vocab = OCRVocab(characters=sorted(set(ch for r in records for ch in r["text"])))

class DirectResizeDataset(Dataset):
    def __init__(self, recs):
        self.recs = recs
        self.cached = []
        for r in recs:
            im = cv2.imread(r["crop_path"], cv2.IMREAD_GRAYSCALE)
            norm = normalize_illumination(im)
            # Resize directly to (32, 96)
            res = cv2.resize(norm, (96, 32))
            normed = (res.astype(np.float32) / 127.5) - 1.0
            lbl = vocab.encode(r["text"])
            self.cached.append((torch.from_numpy(normed).unsqueeze(0).float(), lbl))
    def __len__(self):
        return len(self.cached)
    def __getitem__(self, i):
        return self.cached[i]

def collate_fn(batch):
    imgs, labels = zip(*batch)
    imgs = torch.stack(imgs, 0)
    in_lens = torch.full((len(imgs),), imgs.shape[3] // 4, dtype=torch.long)
    tgt_lens = torch.tensor([len(l) for l in labels], dtype=torch.long)
    flat_targets = torch.tensor([idx for l in labels for idx in l], dtype=torch.long)
    return imgs, flat_targets, in_lens, tgt_lens

ds = DirectResizeDataset(records)
loader = DataLoader(ds, batch_size=16, shuffle=True, collate_fn=collate_fn)

model = CRNN(img_h=32, in_channels=1, num_classes=len(vocab), hidden_size=128, num_lstm_layers=2)
with torch.no_grad():
    model.fc.bias.data[vocab.blank_idx] = -3.0

criterion = nn.CTCLoss(blank=vocab.blank_idx, zero_infinity=True)
optimizer = optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.003, total_steps=20 * len(loader), pct_start=0.1)

print("Starting direct resize test (96x32)...")
for epoch in range(1, 21):
    model.train()
    tot_loss = 0.0
    for imgs, targets, in_lens, tgt_lens in loader:
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits.log_softmax(2), targets, in_lens, tgt_lens)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        scheduler.step()
        tot_loss += loss.item()
    if epoch % 5 == 0 or epoch == 20:
        model.eval()
        correct = 0
        with torch.no_grad():
            for i in range(min(50, len(ds))):
                img, lbl = ds[i]
                l = model(img.unsqueeze(0))
                dec = model.decode_greedy(l, blank_idx=vocab.blank_idx)
                pred = vocab.decode(dec[0])
                gt = records[i]["text"]
                if pred.strip() == gt.strip():
                    correct += 1
        print(f"Epoch {epoch:02d} | Loss: {tot_loss/len(loader):.4f} | Accuracy on 50 samples: {correct}/50 ({correct*2}%)")
        # Print first 3
        for i in range(3):
            img, lbl = ds[i]
            l = model(img.unsqueeze(0))
            dec = model.decode_greedy(l, blank_idx=vocab.blank_idx)
            print(f"  Sample {i}: Pred='{vocab.decode(dec[0])}' | GT='{records[i]['text']}'")
