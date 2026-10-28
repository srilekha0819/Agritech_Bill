import sys
import torch
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))
from models.ocr_crnn import CRNN
from models.vocab import OCRVocab
from scripts.train_digits import prepare_crop, load_digit_records

records = load_digit_records()
vocab = OCRVocab()
vocab.load("saved_models/ocr_digits_vocab.json")

ckpt = torch.load("saved_models/ocr_digits.pth", map_location="cpu")
model = CRNN(img_h=32, in_channels=1, num_classes=len(vocab), hidden_size=128, num_lstm_layers=2)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

for r in records[:15]:
    canvas = prepare_crop(r["crop_path"])
    tensor = torch.from_numpy((canvas.astype(float)/127.5)-1.0).unsqueeze(0).unsqueeze(0).float()
    with torch.no_grad():
        logits = model(tensor)
        greedy = model.decode_greedy(logits, blank_idx=vocab.blank_idx)
        beam = model.decode_beam_search(logits, beam_width=5, blank_idx=vocab.blank_idx)
        p_greedy = vocab.decode(greedy[0])
        p_beam = vocab.decode(beam[0])
    print(f"GT: {r['text']:12s} | Greedy: {p_greedy:12s} | Beam: {p_beam:12s} | Field: {r['field_name']}")
