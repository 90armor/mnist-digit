# Component Guide — mnist-digit

This guide describes every component in the project, its contract, and how components connect.

---

## Component Map

```
config.py
  └── DATA_DIR, MODEL_PATH, TrainConfig
        ├── imported by train.py
        └── imported by test.py

train.py
  ├── build_model(cfg, augmentation) → keras.Model
  └── main()  [CLI entry-point]
        └── writes: digit_model.keras, train_data/*.npy, training_curves.png

test.py
  ├── preprocess_mnist_array(arr)        → (1,28,28,1) tensor
  ├── preprocess_file_image(image_input) → (1,28,28,1) tensor
  ├── predict(model, image_input, ...)   → int
  ├── sanity_check(model, n_samples)
  ├── test_single_local_sample(model, index)
  ├── load_local_test_data()             → (x_test, y_test)
  └── main()  [CLI entry-point]
        └── reads: digit_model.keras, train_data/*.npy

tests/
  └── test_preprocess.py   [pytest suite for test.py preprocessing]
```

---

## `config.py`

Shared constants and hyperparameter configuration. **Both `train.py` and `test.py` import from here.** This is the single source of truth for paths and training settings.

### `DATA_DIR: Path`
- Default: `./train_data`
- Directory where `train.py` writes `.npy` files and `test.py` reads them.

### `MODEL_PATH: Path`
- Default: `./digit_model.keras`
- Path to the saved Keras model artifact.

### `class TrainConfig`
Dataclass holding all training hyperparameters. Pass an instance to `build_model()` and `model.fit()`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `learning_rate` | float | `1e-3` | Initial Adam learning rate |
| `batch_size` | int | `128` | Mini-batch size during training |
| `epochs` | int | `30` | Maximum training epochs (EarlyStopping may stop earlier) |
| `dropout_conv` | float | `0.25` | Dropout rate after each conv block |
| `dropout_dense` | float | `0.50` | Dropout rate before the output layer |
| `filters` | tuple | `(32, 64, 128)` | Conv filter counts per block (3 blocks) |
| `dense_units` | int | `256` | Units in the fully-connected head |

**To run a hyperparameter experiment**, override fields:
```python
from config import TrainConfig
cfg = TrainConfig(learning_rate=5e-4, epochs=50, dropout_dense=0.4)
```

---

## `train.py`

Training pipeline. Designed to be run as a script; safe to import without side effects.

### `build_model(cfg, input_shape, num_classes, augmentation) → keras.Model`

Constructs the CNN from a `TrainConfig`. The `augmentation` parameter is optional — pass `None` to get a model without the augmentation head (useful for transfer or fine-tuning).

**Architecture summary:**
```
Input (28, 28, 1)
  [augmentation layer — only active during training]
  Conv(f[0])×2 → BN → MaxPool → Dropout(dropout_conv)
  Conv(f[1])×2 → BN → MaxPool → Dropout(dropout_conv)
  Conv(f[2])×1 → BN → Dropout(dropout_conv)
  GlobalAveragePooling2D
  Dense(dense_units) → BN → Dropout(dropout_dense)
  Dense(10, softmax)
```

### `main()`

Full training pipeline, executed when the script is run directly:

1. Load MNIST via Keras cache.
2. Save raw `.npy` files to `DATA_DIR` (skipped if they already exist).
3. Preprocess: normalise to `[0,1]`, expand channel dim, one-hot encode labels.
4. Build augmentation sub-model and pass to `build_model`.
5. Compile with Adam + categorical crossentropy.
6. Train with `ModelCheckpoint`, `ReduceLROnPlateau`, and `EarlyStopping`.
7. Evaluate the in-memory model (best weights already restored by `EarlyStopping`).
8. Save training curve plot to `training_curves.png`.

**Callbacks:**
| Callback | Monitors | Behaviour |
|----------|----------|-----------|
| `ModelCheckpoint` | `val_accuracy` | Saves best model to `MODEL_PATH` |
| `ReduceLROnPlateau` | `val_loss` | Halves LR after 3 stagnant epochs; floor `1e-6` |
| `EarlyStopping` | `val_accuracy` | Stops after 8 stagnant epochs; restores best weights |

---

## `test.py`

Inference and evaluation script. Safe to import (all execution is inside `main()` or explicit function calls).

### `preprocess_mnist_array(arr: np.ndarray) → np.ndarray`

**Contract:** Takes a raw `uint8` 28×28 MNIST array. Divides by 255 and adds batch + channel dimensions.

- Input shape: `(28, 28)` uint8
- Output shape: `(1, 28, 28, 1)` float32, values in `[0, 1]`
- No denoising, cropping, or padding — MNIST data is already clean and correctly sized.

Use this path for: `sanity_check`, `test_single_local_sample`, any caller that already has a raw MNIST array.

### `preprocess_file_image(image_input, target_size=28) → np.ndarray`

**Contract:** Takes a file path (`str` or `Path`) or a `PIL.Image` of any resolution and mode.

Processing steps:
1. Convert to greyscale (`L` mode)
2. Auto-invert if mean pixel > 127 (corrects white-background images)
3. Median filter (3×3) for noise removal
4. Tight-crop to bounding box of significant pixels (threshold: 15% of max)
5. Pad to square with proportional margin (`max(4, side // 7)` px)
6. Resize to `target_size × target_size` with LANCZOS resampling
7. Normalise to `[0, 1]` + min-max contrast stretch

- Output shape: `(1, target_size, target_size, 1)` float32, values in `[0, 1]`

Use this path for: user-supplied image files, any arbitrary-resolution input.

### `predict(model, image_input, show, true_label) → int`

Dispatches to the correct preprocessing function based on `image_input` type, runs inference, and prints per-class probability bars to the terminal. If `show=True`, opens a Matplotlib window with the preprocessed patch and prediction.

**Type dispatch:**
- `np.ndarray` → `preprocess_mnist_array`
- `str / Path / PIL.Image` → `preprocess_file_image`

### `load_local_test_data() → (np.ndarray, np.ndarray)`

Loads `test_images.npy` and `test_labels.npy` from `DATA_DIR`.

**Raises:** `FileNotFoundError` if the files are missing (with a message directing the user to run `train.py`).

### `sanity_check(model, n_samples=2000)`

Runs batch inference on the first `n_samples` from local MNIST test data, reports accuracy, and shows a 2×5 grid of sample predictions.

Uses `preprocess_mnist_array` for normalization — consistent with `predict()`.

### `test_single_local_sample(model, index, show=True)`

Runs inference on a single sample from the local test split by index.

**Raises:** `FileNotFoundError` if data is missing; `IndexError` if index is out of range.

### CLI flags

| Flag | Description |
|------|-------------|
| `image` (positional, optional) | Path to an image file to classify |
| `--show` | Display preprocessed patch and result in a Matplotlib window |
| `--model PATH` | Override the model file path (default: `digit_model.keras`) |
| `--sample N` | Test sample index `N` from local test data |
| `--n N` | Number of samples for sanity check (default: 2000) |

---

## `tests/test_preprocess.py`

Pytest suite covering `preprocess_mnist_array`, `preprocess_file_image`, and `load_local_test_data`.

Run from the project root:
```bash
pytest tests/
```

The `conftest.py` at the project root adds the project root to `sys.path` so imports from `test.py` and `config.py` resolve correctly.

### Test coverage

| Function | Tests |
|----------|-------|
| `preprocess_mnist_array` | Output shape, dtype, normalisation, zero input, relative ordering |
| `preprocess_file_image` | Output shape/dtype/range, RGB input, file path (Path + str), custom target size |
| `load_local_test_data` | Raises `FileNotFoundError` when data is missing |

---

## Data Flow Between Components

```
python train.py
  → config.DATA_DIR/*.npy      (raw MNIST arrays, saved once)
  → config.MODEL_PATH          (best checkpoint via ModelCheckpoint)
  → training_curves.png

python test.py [image]
  ← config.MODEL_PATH          (loaded at startup)
  ← config.DATA_DIR/*.npy      (loaded only in sanity_check / test_single_local_sample)
```

---

## Adding a New Hyperparameter

1. Add the field to `TrainConfig` in [config.py](config.py).
2. Pass `cfg.<field>` where the value is used in `build_model()` or `main()` in [train.py](train.py).
3. No changes needed in `test.py` — inference is hyperparameter-independent.
