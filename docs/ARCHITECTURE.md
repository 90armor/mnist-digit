# ARCHITECTURE.md — mnist-digit

**Branch:** HigherResolution  
**Last updated:** 2026-06-11

---

## 1. Overview

`mnist-digit` is a local CLI machine learning tool that trains a Convolutional Neural Network (CNN) on MNIST and runs inference on arbitrary-resolution digit images. There is no server, no database, and no network dependency at runtime.

The project is structured as a **train → artifact → infer** pipeline across two entry-point scripts backed by a shared config module.

---

## 2. Module Map

```
mnist-digit/
├── src/
│   ├── config.py              # Single source of truth: paths, TrainConfig dataclass
│   ├── train.py               # Training pipeline entry point
│   ├── test.py                # Inference / evaluation entry point
│   ├── conftest.py            # Adds src/ to sys.path for pytest
│   └── tests/
│       └── test_preprocess.py # 13 pytest tests for preprocessing functions
│
├── docs/                      # Project documentation
├── .ai/                       # Agent instructions and code conventions
│
├── digit_model.keras          # Saved model artifact (written by train.py)
├── training_curves.png        # Accuracy & loss plot (written by train.py)
│
├── train_data/                # Cached MNIST arrays (written once by train.py)
│   ├── train_images.npy       # (60 000 × 28 × 28) uint8
│   ├── train_labels.npy       # (60 000,) int
│   ├── test_images.npy        # (10 000 × 28 × 28) uint8
│   └── test_labels.npy        # (10 000,) int
│
├── test_data/                 # User-supplied inference images (arbitrary PNG)
└── requirements.txt           # Pinned runtime + dev dependencies
```

---

## 3. Dependency Graph

```
config.py
    ↑           ↑
train.py      test.py
```

`train.py` and `test.py` are independent entry points that share only `config.py`. Neither imports the other.

---

## 4. Tech Stack

| Category | Technology |
|---|---|
| Language | Python 3 |
| Deep learning | TensorFlow / Keras |
| Numerical computing | NumPy |
| Image processing | Pillow (PIL) |
| Visualisation | Matplotlib |
| Dataset | MNIST via `keras.datasets.mnist` |
| Model serialisation | Keras native (`.keras`) |
| CLI parsing | `argparse` (stdlib) |
| Logging | `logging` (stdlib) |
| Testing | pytest |

---

## 5. Shared Config Layer (`config.py`)

All paths and training hyperparameters are defined once here. Both scripts import from this module — there are no duplicate constants.

```python
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
```

---

## 6. Training Pipeline (`train.py`)

### Entry point

```
python train.py
```

### Flow

```
main()
  │
  ├─ 1. Set random seeds (tf 42, np 42)
  ├─ 2. Load MNIST via keras.datasets.mnist.load_data()
  ├─ 3. Cache raw arrays to DATA_DIR/*.npy       ← skipped if files already exist
  ├─ 4. Normalise pixels [0,1]; add channel dim  → (N, 28, 28, 1)
  ├─ 5. One-hot encode labels                    → (N, 10)
  ├─ 6. Build augmentation Sequential model
  ├─ 7. build_model(cfg, augmentation=augmentation)
  ├─ 8. Compile: Adam(lr), categorical crossentropy
  ├─ 9. model.fit — up to 30 epochs, batch 128, three callbacks
  │       ├─ ModelCheckpoint  → saves best val_accuracy to MODEL_PATH
  │       ├─ ReduceLROnPlateau → halves LR after 3 stagnant val_loss epochs
  │       └─ EarlyStopping    → stops after 8 epochs; restores best weights
  ├─ 10. Evaluate in-memory model on test split  (no redundant reload)
  └─ 11. Save training_curves.png
```

### `build_model(cfg, input_shape, num_classes, augmentation)`

Augmentation is passed as an explicit parameter — it is **not** captured from outer scope.

```
Input (28, 28, 1)
│
├── Augmentation (RandomRotation 10%, RandomZoom 10%, RandomTranslation 10%)  [train only]
│
├── Conv2D(32, 3×3, same, relu) → BatchNorm
├── Conv2D(32, 3×3, same, relu) → BatchNorm → MaxPool(2×2) → Dropout(0.25)
│
├── Conv2D(64, 3×3, same, relu) → BatchNorm
├── Conv2D(64, 3×3, same, relu) → BatchNorm → MaxPool(2×2) → Dropout(0.25)
│
├── Conv2D(128, 3×3, same, relu) → BatchNorm → Dropout(0.25)
│
├── GlobalAveragePooling2D
├── Dense(256, relu) → BatchNorm → Dropout(0.5)
│
└── Dense(10, softmax)   →   class probabilities (digits 0–9)
```

---

## 7. Inference Pipeline (`test.py`)

### Entry point

| Mode | Command |
|---|---|
| Predict on a file | `python test.py image.png [--show]` |
| Test one MNIST sample | `python test.py --sample 3 [--show]` |
| Bulk sanity check | `python test.py [--n 2000]` |

### Flow

```
main()
  │
  ├─ Parse CLI args
  ├─ Load digit_model.keras from MODEL_PATH
  └─ Route:
       ├─ file path  → predict(model, path)   → preprocess_file_image()
       ├─ --sample N → test_single_local_sample()  → preprocess_mnist_array()
       └─ default    → sanity_check()          → preprocess_mnist_array()
```

### Preprocessing contract — two explicit functions

#### `preprocess_mnist_array(arr: np.ndarray) → np.ndarray`

For raw 28×28 uint8 MNIST arrays (sanity check, sample mode):

```
uint8 array (28, 28)
  → float32 / 255.0
  → shape: (1, 28, 28, 1)
```

#### `preprocess_file_image(image_input, target_size=28) → np.ndarray`

For arbitrary-resolution file paths or PIL Images:

```
Any-resolution image (any format)
  → Grayscale (L mode)
  → Auto-invert if mean pixel > _INVERT_THRESHOLD (127)
  → Median filter (_MEDIAN_FILTER_SIZE = 3) for denoising
  → Tight-crop bounding box (_CROP_THRESHOLD_RATIO = 0.15)
  → Pad to square with margin max(_MIN_MARGIN_PX=4, side/_MARGIN_DIVISOR=7)
  → Resize to 28×28 (LANCZOS)
  → Normalise [0,1] + min-max contrast stretch
  → shape: (1, 28, 28, 1)
```

Named constants replace all magic numbers: `_CROP_THRESHOLD_RATIO`, `_INVERT_THRESHOLD`, `_MARGIN_DIVISOR`, `_MIN_MARGIN_PX`, `_MEDIAN_FILTER_SIZE`, `_BAR_WIDTH_CHARS`.

---

## 8. State Management

All state is file-based. There is no in-memory model cache; `test.py` loads the model fresh on each invocation.

| Artifact | Written by | Read by |
|---|---|---|
| `digit_model.keras` | `train.py` (ModelCheckpoint) | `test.py` |
| `train_data/*.npy` | `train.py` (once; skipped if exists) | `test.py` (sample / sanity check) |
| `training_curves.png` | `train.py` | User |

---

## 9. Error Handling

Library functions raise standard Python exceptions; only `main()` calls `sys.exit()`.

| Function | Raises |
|---|---|
| `load_local_test_data()` | `FileNotFoundError` if `DATA_DIR` is missing |
| `test_single_local_sample()` | `IndexError` if index is out of range |

```python
# main() boundary
except (FileNotFoundError, IndexError) as exc:
    sys.exit(f"Error: {exc}")
```

---

## 10. Logging

Both scripts use `logging` (stdlib). No `print()` calls remain.

```python
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)
```

---

## 11. Testing

```bash
pytest tests/
```

`tests/test_preprocess.py` contains 13 tests covering:
- `preprocess_mnist_array` — shape, dtype, value range
- `preprocess_file_image` — grayscale, inversion, crop, resize, contrast stretch
- `load_local_test_data` — missing-data error path

`conftest.py` adds the project root to `sys.path` so `import config`, `import test` resolve without installation.

---

## 12. Performance Design Decisions

| Decision | Rationale |
|---|---|
| GlobalAveragePooling2D | Reduces parameter count vs Flatten; lower overfitting risk |
| BatchNorm after every conv | Stabilises training; permits higher LR |
| Dropout 0.25 (conv) / 0.5 (dense) | Regularisation at the two highest-risk layers |
| Data augmentation in model graph | Zero preprocessing overhead at inference for MNIST-sized inputs |
| LANCZOS resampling | Highest quality for downsampling arbitrary → 28×28 |
| EarlyStopping + restore_best_weights | Avoids unnecessary epochs and removes the need for a post-training reload |
| .npy existence check before save | Avoids 200 MB+ disk write on repeated training runs |

---

## 13. Security Posture

The attack surface is limited to a local CLI with no network listeners.

| Surface | Note |
|---|---|
| Arbitrary file reads | Pillow `Image.MAX_IMAGE_PIXELS` limits decompression bomb risk |
| Model deserialisation | `load_model` is safe for `.keras` files from trusted sources; Lambda layers in untrusted models could execute arbitrary Python |
| No credentials or secrets | None present in codebase |

---

## 14. Planned Improvements

| Priority | Item |
|---|---|
| High | Multi-digit detection (referenced in prior commit history) |
| High | Higher-resolution native input support (active branch goal) |
| Medium | ONNX / TFLite export for edge/mobile deployment |
| Medium | REST API or web UI wrapper (FastAPI / Flask) |
| Low | Per-class metrics and confusion matrix in sanity check |
| Low | CLI flags for all `TrainConfig` fields |
