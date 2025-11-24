# cnn/config.py
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
ASSETS_DIR = ROOT / "assets"
WEIGHTS_PATH = MODELS_DIR / "weights.pt"

IMG_SIZE = 64
BATCH_SIZE = 128
LR = 1e-3
EPOCHS = 10
NUM_WORKERS = 2

CLASSES = [
    "airplane","automobile","bird","cat","deer",
    "dog","frog","horse","ship","truck"
]
MEAN = (0.4914, 0.4822, 0.4465)
STD  = (0.2470, 0.2435, 0.2616)
