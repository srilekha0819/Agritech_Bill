import json
import os
from typing import List, Dict, Optional

class OCRVocab:
    """Character-level vocabulary for multilingual CRNN CTC OCR."""
    def __init__(self, characters: Optional[List[str]] = None):
        self.pad_token = "<PAD>"
        self.blank_token = "<BLANK>"  # CTC blank token must be index 0
        self.unk_token = "<UNK>"
        
        # Telugu characters: Vowels, consonants, matras, virama, signs
        # Unicode range 0C00 - 0C7F
        telugu_chars = [
            'ఁ', 'ం', 'ః',
            'అ', 'ఆ', 'ఇ', 'ఈ', 'ఉ', 'ఊ', 'ఋ', 'ఌ', 'ఎ', 'ఏ', 'ఐ', 'ఒ', 'ఓ', 'ఔ',
            'క', 'ఖ', 'గ', 'ఘ', 'ఙ',
            'చ', 'ఛ', 'జ', 'ఝ', 'ఞ',
            'ట', 'ఠ', 'డ', 'ఢ', 'ణ',
            'త', 'థ', 'ద', 'ధ', 'న',
            'ప', 'ఫ', 'బ', 'భ', 'మ',
            'య', 'ర', 'ఱ', 'ల', 'ళ', 'వ', 'శ', 'ష', 'స', 'హ',
            'ా', 'ి', 'ీ', 'ు', 'ూ', 'ృ', 'ౄ', 'ె', 'ే', 'ై', 'ొ', 'ో', 'ౌ', '్',
            'ౕ', 'ౖ', 'ౘ', 'ౙ', 'ౚ', 'ౠ', 'ౡ', 'ౢ', 'ౣ',
            '౦', '౧', '౨', '౩', '౪', '౫', '౬', '౭', '౮', '౯' # Telugu digits
        ]
        
        # English alphabet
        english_chars = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
        
        # Digits
        digits = list("0123456789")
        
        # Punctuation & symbols common in bills
        symbols = list(" .,:;/-+=%()[]{}'\"*#\\!?&|₹$@")
        
        if characters is None:
            character_set = set(digits + english_chars + symbols + telugu_chars)
        else:
            character_set = set(characters)
        character_set.difference_update({self.pad_token, self.blank_token, self.unk_token})
        all_chars = [self.blank_token, self.unk_token] + sorted(character_set)
        
        self.char2idx: Dict[str, int] = {ch: idx for idx, ch in enumerate(all_chars)}
        self.idx2char: Dict[int, str] = {idx: ch for idx, ch in enumerate(all_chars)}
        self.blank_idx = 0
        self.unk_idx = self.char2idx.get(self.unk_token, 1)

    def encode(self, text: str) -> List[int]:
        """Convert string to list of token IDs."""
        return [self.char2idx.get(ch, self.unk_idx) for ch in text]

    def decode(self, indices: List[int]) -> str:
        """Convert list of token IDs back to string, skipping blank and UNK."""
        chars = []
        for idx in indices:
            if idx == self.blank_idx:
                continue
            char = self.idx2char.get(idx, "")
            if char and char not in (self.blank_token, self.unk_token, self.pad_token):
                chars.append(char)
        return "".join(chars)

    def contains(self, ch: str) -> bool:
        """Return True if the character is in the vocabulary (not UNK)."""
        return ch in self.char2idx and self.char2idx[ch] != self.unk_idx

    def __len__(self) -> int:
        return len(self.char2idx)

    def save(self, filepath: str):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({
                "char2idx": self.char2idx,
                "idx2char": {str(k): v for k, v in self.idx2char.items()}
            }, f, ensure_ascii=False, indent=2)

    def load(self, filepath: str):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.char2idx = data["char2idx"]
            self.idx2char = {int(k): v for k, v in data["idx2char"].items()}


class Seq2SeqVocab:
    """Word and sub-token vocabulary for neural translation."""
    def __init__(self, name: str = "vocab"):
        self.name = name
        self.pad_token = "<PAD>"
        self.sos_token = "<SOS>"
        self.eos_token = "<EOS>"
        self.unk_token = "<UNK>"
        
        self.word2idx: Dict[str, int] = {
            self.pad_token: 0,
            self.sos_token: 1,
            self.eos_token: 2,
            self.unk_token: 3
        }
        self.idx2word: Dict[int, str] = {
            0: self.pad_token,
            1: self.sos_token,
            2: self.eos_token,
            3: self.unk_token
        }
        self.word_count: Dict[str, int] = {}
        self.n_words = 4
        self.pad_idx = 0
        self.sos_idx = 1
        self.eos_idx = 2
        self.unk_idx = 3

    def add_word(self, word: str):
        if word not in self.word2idx:
            self.word2idx[word] = self.n_words
            self.idx2word[self.n_words] = word
            self.word_count[word] = 1
            self.n_words += 1
        else:
            self.word_count[word] += 1

    def add_sentence(self, sentence: str):
        for token in sentence.strip().split():
            self.add_word(token)

    def encode(self, sentence: str, add_sos_eos: bool = True) -> List[int]:
        tokens = sentence.strip().split()
        indices = [self.word2idx.get(w, self.unk_idx) for w in tokens]
        if add_sos_eos:
            return [self.sos_idx] + indices + [self.eos_idx]
        return indices

    def decode(self, indices: List[int], remove_special: bool = True) -> str:
        words = []
        for idx in indices:
            if remove_special and idx in (self.pad_idx, self.sos_idx, self.eos_idx):
                if idx == self.eos_idx:
                    break
                continue
            word = self.idx2word.get(idx, self.unk_token)
            words.append(word)
        return " ".join(words)

    def __len__(self) -> int:
        return self.n_words

    def save(self, filepath: str):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({
                "word2idx": self.word2idx,
                "idx2word": {str(k): v for k, v in self.idx2word.items()},
                "n_words": self.n_words
            }, f, ensure_ascii=False, indent=2)

    def load(self, filepath: str):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.word2idx = data["word2idx"]
            self.idx2word = {int(k): v for k, v in data["idx2word"].items()}
            self.n_words = data["n_words"]
