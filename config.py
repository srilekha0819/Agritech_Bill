import os
import torch

# Load environment variables from .env file if python-dotenv is installed
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVED_MODELS_DIR = os.getenv("SAVED_MODELS_DIR", os.path.join(BASE_DIR, "saved_models"))
UPLOADS_DIR = os.getenv("UPLOADS_DIR", os.path.join(BASE_DIR, "uploads"))
OCR_ANNOTATIONS_PATH = os.path.join(BASE_DIR, "data_generation", "ocr_annotations.jsonl")
OCR_DATASET_MANIFEST_PATH = os.path.join(BASE_DIR, "ocr_dataset", "current", "manifest.jsonl")
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

# Server configuration
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "5000"))
DEBUG = os.getenv("FLASK_DEBUG", "false").lower() in ("true", "1", "yes")
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key-langbill-ai")
MAX_CONTENT_LENGTH_MB = int(os.getenv("MAX_CONTENT_LENGTH_MB", "25"))
MAX_CONTENT_LENGTH = MAX_CONTENT_LENGTH_MB * 1024 * 1024

# Device configuration
def _get_device() -> torch.device:
    device_override = os.getenv("DEVICE", "auto").lower().strip()
    if device_override == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_override == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEVICE = _get_device()

# Model paths
OCR_MODEL_PATH = os.path.join(SAVED_MODELS_DIR, "ocr_crnn.pth")
OCR_VOCAB_PATH = os.path.join(SAVED_MODELS_DIR, "ocr_vocab.json")
TRANSLATOR_MODEL_PATH = os.path.join(SAVED_MODELS_DIR, "translator_seq2seq.pth")
TRANSLATOR_SRC_VOCAB_PATH = os.path.join(SAVED_MODELS_DIR, "translator_src_vocab.json")
TRANSLATOR_TGT_VOCAB_PATH = os.path.join(SAVED_MODELS_DIR, "translator_tgt_vocab.json")

# OCR Hyperparameters
OCR_IMG_H = 32         # Standard height for CRNN
OCR_IMG_W = 320        # 40% wider than 256: better char resolution, still CPU-trainable
OCR_IN_CHANNELS = 1   # Grayscale
OCR_HIDDEN_SIZE = 192  # More capacity than 128, less than 256
OCR_NUM_LAYERS = 2
OCR_DROPOUT = 0.2
OCR_BATCH_SIZE = 24    # Good balance for 320-width images on CPU
OCR_EPOCHS = 50       # Practical for CPU (~2-4 hours)
OCR_LR = 0.001

# Translation Hyperparameters
TRANS_EMBED_DIM = 128
TRANS_HIDDEN_SIZE = 256
TRANS_NUM_LAYERS = 2
TRANS_DROPOUT = 0.2
TRANS_BATCH_SIZE = 32
TRANS_EPOCHS = 8
TRANS_LR = 0.0015
TRANS_MAX_LEN = 80

# Special Tokens
PAD_TOKEN = "<PAD>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"
BLANK_TOKEN = "<BLANK>"  # For CTC
