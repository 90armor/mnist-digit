from pathlib import Path
from dataclasses import dataclass

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR    = _PROJECT_ROOT / "train_data"
MODEL_PATH  = _PROJECT_ROOT / "digit_model.keras"
CURVES_PATH = _PROJECT_ROOT / "training_curves.png"


@dataclass
class TrainConfig:
    learning_rate: float = 1e-3
    batch_size:    int   = 128
    epochs:        int   = 30
    dropout_conv:  float = 0.25
    dropout_dense: float = 0.50
    filters:       tuple = (32, 64, 128)
    dense_units:   int   = 256
