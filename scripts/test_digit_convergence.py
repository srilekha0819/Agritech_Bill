import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from difflib import SequenceMatcher
from pathlib import Path
import sys

sys.path.insert(0, str(Path(".").resolve()))
from models.ocr_crnn import CRNN
from models.vocab import OCRVocab
from scripts.train_digits import DigitsCropDataset, collate_fn, load_digit_records

records = load_digit_records()
vocab = OCRVocab(characters=sorted(set(ch for r in records for ch in r["text"])))

# Let's inspect
print(f"Total digit records: {len(records)}, Vocab: {len(vocab)}")

train_ds = DigitsCropDataset(records, vocab, augment=True)
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collate_fn)

model = CRNN(img_h=32, in_channels=1, num_classes=len(vocab), hidden_size=128, num_lstm_layers=2)
# Set blank bias to -4.0 so it doesn't collapse to blank initially
with torch.no_grad():
    model.fc.bias.data[vocab.blank_idx] = -4.0

criterion = nn.CTCLoss(blank=vocab.blank_idx, zero_infinity=True)
optimizer = optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
total_steps = 25 * len(train_loader)
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.002, total_steps=total_steps, pct_start=0.15)

for epoch in range(1, 26):
    model.train()
    total_loss = 0.0
    for imgs, targets, in_lens, tgt_lens in train_loader:
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits.log_softmax(2), targets, in_lens, tgt_lens)
        if torch.isfinite(loss):
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
    if epoch % 5 == 0 or epoch == 25:
        avg_loss = total_loss / len(train_loader)
        # Check a sample prediction
        model.eval()
        with torch.no_grad():
            sample_img, sample_lbl = train_ds[0]
            l = model(sample_img.unsqueeze(0))
            dec = model.decode_greedy(l, blank_idx=vocab.blank_idx)
            pred = vocab.decode(dec[0])
            gt = records[0]["text"]
        print(f"Epoch {epoch:02d} | Loss: {avg_loss:.4f} | Sample Pred: '{pred}' (GT: '{gt}')")
