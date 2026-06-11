from pathlib import Path
from dataclasses import dataclass

DATA_DIR   = Path("./train_data")
MODEL_PATH = Path("digit_model.keras")


@dataclass
class TrainConfig:
    learning_rate: float = 1e-3
    batch_size:    int   = 128
    epochs:        int   = 30
    dropout_conv:  float = 0.25
    dropout_dense: float = 0.50
    filters:       tuple = (32, 64, 128)
    dense_units:   int   = 256
