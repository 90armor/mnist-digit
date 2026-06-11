# PROJECT_OVERVIEW.md

## 1. Project Overview

**mnist-digit** is a command-line machine learning project that trains and deploys a Convolutional Neural Network (CNN) to recognize handwritten digits (0–9). The model is trained on the MNIST benchmark dataset and can subsequently classify digit images of any resolution supplied by the user.

The project is split into two self-contained scripts:

| Script | Responsibility |
|---|---|
| `train.py` | Download MNIST, preprocess data, build and train the CNN, save model artifact |
| `test.py` | Load saved model, preprocess arbitrary-resolution images, run inference |

This is a standalone, offline ML research/tool project with no web server, no database, and no external API dependencies at runtime.

---

## 2. Core Features

- **MNIST-based CNN training** — trains a deep CNN from scratch on 60,000 labeled digit images.
- **Data persistence** — MNIST data is saved locally as `.npy` files so training can be re-run without re-downloading.
- **Data augmentation** — random rotation, zoom, and translation applied during training to improve generalisation.
- **Arbitrary-resolution inference** — `test.py` accepts images of any resolution and pixel mode, normalising them to the 28×28 input the model expects.
- **Auto-invert detection** — automatically detects and corrects images with white backgrounds (MNIST uses white-digit/black-background convention).
- **Confidence reporting** — prints per-class probabilities as a bar chart in the terminal.
- **Sanity check mode** — evaluates the model against the locally cached MNIST test split and shows 10 random sample predictions.
- **Training curve visualisation** — saves a `training_curves.png` plot of accuracy and loss over epochs.
- **Reproducibility** — fixed random seeds (`tf.random.set_seed(42)`, `np.random.seed(42)`) ensure consistent training runs.

---

## 3. Tech Stack

| Category | Technology |
|---|---|
| Language | Python 3 |
| Deep Learning | TensorFlow / Keras (`tensorflow`) |
| Numerical computing | NumPy |
| Image processing | Pillow (PIL) |
| Visualisation | Matplotlib |
| Dataset | MNIST (via `keras.datasets.mnist`) |
| Model serialisation | Keras native format (`.keras`) |
| CLI parsing | `argparse` (stdlib) |

No web framework, no containerisation config, and no package manifest (`requirements.txt`) is present in the repository.

---

## 4. Architecture Overview

This project follows a classic **train → artifact → infer** ML pipeline architecture.

```
┌──────────────────────────────────────────────────────────────┐
│                        TRAINING PHASE                        │
│                                                              │
│  MNIST Dataset ──► Preprocess ──► Augment ──► CNN Model      │
│                                                  │           │
│                                           digit_model.keras  │
│                                          training_curves.png │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│                       INFERENCE PHASE                        │
│                                                              │
│  Any-resolution image                                        │
│        │                                                     │
│        ▼                                                     │
│  Preprocess (greyscale → invert → denoise → crop → pad       │
│              → 28×28 resize → normalise)                     │
│        │                                                     │
│        ▼                                                     │
│  digit_model.keras ──► Softmax(10) ──► Predicted digit       │
│                                      + Confidence %          │
└──────────────────────────────────────────────────────────────┘
```

There is no client-server separation. Both phases run locally on the same machine via the Python CLI.

---

## 5. Folder Structure

```
mnist-digit/
├── train.py                   # Training pipeline script
├── test.py                    # Inference / evaluation script
├── digit_model.keras          # Saved best model (produced by train.py)
├── training_curves.png        # Accuracy & loss plot (produced by train.py)
│
├── train_data/                # Locally cached MNIST data (produced by train.py)
│   ├── train_images.npy       # (60,000 × 28 × 28) uint8
│   ├── train_labels.npy       # (60,000,) int
│   ├── test_images.npy        # (10,000 × 28 × 28) uint8
│   └── test_labels.npy        # (10,000,) int
│
└── test_data/                 # Custom test images (arbitrary resolution PNGs)
    ├── image1.png
    ├── image2.png
    …
    └── image9.png
```

> **Note:** `test.py` currently references `./data/` as `DATA_DIR` while `train.py` saves to `./train_data/`. These paths are inconsistent; running the default sanity-check mode in `test.py` will fail unless the path is corrected or data is placed in `./data/`.

---

## 6. Main Application Flow

### Training (`python train.py`)

1. Set random seeds for reproducibility.
2. Load MNIST via `keras.datasets.mnist.load_data()`.
3. Persist raw arrays to `./train_data/*.npy` and 10 sample PNGs for visual inspection.
4. Normalise pixel values to `[0, 1]` and add a channel dimension `(N, 28, 28, 1)`.
5. One-hot encode labels to `(N, 10)`.
6. Define a `keras.Sequential` data augmentation sub-model.
7. Build the CNN (see §8).
8. Compile with Adam (`lr=1e-3`) and categorical crossentropy.
9. Train for up to 30 epochs with batch size 128, using three callbacks.
10. Reload the best checkpoint and evaluate on the test split.
11. Save training curve plots to `training_curves.png`.

### Inference (`python test.py [image] [flags]`)

| Mode | Command |
|---|---|
| Predict on a file | `python test.py image.png [--show]` |
| Test one local sample | `python test.py --sample 3 [--show]` |
| Sanity check (default) | `python test.py [--n 2000]` |

1. Parse CLI arguments.
2. Load `digit_model.keras` from disk.
3. Route to: file prediction, single local-sample test, or bulk sanity check.
4. In file prediction: open the image, run `preprocess_image`, call `model.predict`.
5. Print per-class probability bar chart; optionally display the preprocessed 28×28 patch with Matplotlib.

---

## 7. State Management

There is no runtime state management in the traditional sense. All state is file-based:

| Artifact | Written by | Read by |
|---|---|---|
| `digit_model.keras` | `train.py` (ModelCheckpoint callback) | `test.py` |
| `train_data/*.npy` | `train.py` | `test.py` (sanity check / sample mode) |
| `training_curves.png` | `train.py` | User (visual inspection) |

The model is loaded fresh on each invocation of `test.py`; there is no persistent server or in-memory model cache.

---

## 8. API / Data Flow

There is no HTTP API. Data flows entirely through local files and in-process function calls.

### CNN Architecture (defined in `train.py: build_model`)

```
Input (28, 28, 1)
│
├── Data Augmentation (RandomRotation 10%, RandomZoom 10%, RandomTranslation 10%)
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
└── Dense(10, softmax)   →  class probabilities (digits 0–9)
```

**Optimizer:** Adam, initial `lr=1e-3`  
**Loss:** Categorical crossentropy  
**Callbacks:**
- `ModelCheckpoint` — saves best `val_accuracy` model to `digit_model.keras`
- `ReduceLROnPlateau` — halves LR on 3 consecutive stagnant `val_loss` epochs (min `1e-6`)
- `EarlyStopping` — stops after 8 epochs without `val_accuracy` improvement; restores best weights

### Preprocessing pipeline (`test.py: preprocess_image`)

```
Raw image (any format, any resolution)
  → Convert to grayscale (L mode)
  → Auto-invert if mean pixel > 127
  → Median filter (3×3) for denoising
  → Tight-crop bounding box (15% relative threshold)
  → Pad to square with proportional margin (max(4, side/7))
  → Resize to 28×28 (LANCZOS resampling)
  → Normalise [0,1] + min-max contrast stretch
  → Shape: (1, 28, 28, 1)
```

---

## 9. UI/UX Direction

The project has **no graphical user interface**. Interaction is entirely via the terminal:

- **Training** outputs epoch-by-epoch Keras progress bars, final accuracy/loss metrics, and saves a PNG plot of training curves.
- **Inference** outputs a formatted terminal table of per-class probabilities using Unicode block characters (`█`) as a bar chart.
- The `--show` flag displays Matplotlib windows for visual inspection of the preprocessed input and predicted label.

The UX is oriented towards ML practitioners and developers running experiments locally, not end users.

---

## 10. Performance Considerations

- **GlobalAveragePooling2D** instead of `Flatten` reduces parameter count and overfitting risk.
- **BatchNormalization** after every convolutional layer stabilises training and allows higher learning rates.
- **Dropout** (0.25 after conv blocks, 0.5 before the output layer) regularises the model.
- **Data augmentation** (rotation, zoom, translation) is applied only during training and baked into the model graph, meaning no preprocessing overhead at inference time for standard MNIST-sized inputs.
- **LANCZOS resampling** during preprocessing provides high-quality downsampling from arbitrary resolutions to 28×28.
- **Batch size 128** balances gradient stability with memory and iteration speed.
- **ReduceLROnPlateau** prevents LR from remaining too high in later training, improving final convergence.
- **EarlyStopping** avoids unnecessary training epochs once the model plateaus.

---

## 11. Security Considerations

As a local CLI tool with no network-facing components, the attack surface is minimal:

- **Arbitrary file reads:** `test.py` opens user-supplied image paths without validation beyond existence checks. Maliciously crafted image files (e.g. ZIP bombs or decompression bombs via Pillow) could cause excessive memory use. Pillow's `Image.MAX_IMAGE_PIXELS` limit provides some protection by default.
- **Model loading:** `keras.models.load_model` deserialises a `.keras` file from disk. Loading a model from an untrusted source could execute arbitrary Python if the model contains custom Lambda layers or objects. Only load models from trusted sources.
- **No user input sanitisation needed** for the core ML pipeline itself, as data flows through NumPy arrays rather than interpreted strings.
- **No credentials, secrets, or tokens** are present in the codebase.

---

## 12. Future Improvements

Based on the current state of the code and git history:

1. **Fix DATA_DIR path mismatch** — `test.py` references `./data/` but `train.py` saves to `./train_data/`. These should be reconciled to a single shared constant.
2. **Add `requirements.txt`** — no dependency manifest exists; contributors must infer requirements from imports.
3. **Multi-digit detection** — a prior git commit references multi-digit recognition (`muilti-digit`), suggesting this is a planned or in-progress capability.
4. **Higher-resolution native support** — the active branch (`HigherResolution`) indicates work to improve handling of high-resolution real-world digit images without quality loss from aggressive downsampling.
5. **Model export** — exporting to ONNX or TensorFlow Lite would enable deployment on mobile or edge devices and allow inference without a full TensorFlow installation.
6. **REST API or web UI** — wrapping `test.py` in a FastAPI or Flask server would make the model accessible to other applications without Python runtime dependencies on the client.
7. **Confusion matrix and per-class metrics** — the sanity check reports aggregate accuracy but no per-class breakdown or confusion matrix.
8. **Automated tests** — no test suite exists; adding `pytest`-based unit tests for the preprocessing pipeline would prevent regressions during refactoring.
9. **Configurable hyperparameters via CLI** — training hyperparameters (learning rate, epochs, batch size, dropout) are hardcoded; exposing them via `argparse` would improve experimentation ergonomics.
