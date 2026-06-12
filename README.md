# mnist-digit

A command-line CNN that recognises handwritten digits (0–9). Trained on MNIST; runs inference on any image — any resolution, any format.

---

## Features

- **CNN training** on 60,000 MNIST images with data augmentation (rotation, zoom, translation)
- **Arbitrary-resolution inference** — images are preprocessed to 28×28 automatically
- **Auto-invert detection** — handles white-background images without manual preprocessing
- **Terminal confidence bars** — per-class probability chart rendered in the terminal
- **Training curve plots** — accuracy and loss visualised per epoch
- **Reproducible** — fixed random seeds (`tf 42`, `np 42`) for consistent results
- **Dockerised** — two-stage image with pre-built wheels; no network access at runtime
- **Test suite** — 13 pytest tests covering the full preprocessing pipeline

---

## Project Structure

```
mnist-digit/
│
├── README.md
│
├── src/
│   ├── config.py              # Shared paths and TrainConfig dataclass
│   ├── train.py               # Training pipeline entry point
│   ├── test.py                # Inference and evaluation entry point
│   ├── conftest.py            # pytest sys.path setup
│   └── tests/
│       └── test_preprocess.py # 13 preprocessing unit tests
│
├── docs/
│   ├── PROJECT_OVERVIEW.md    # Full project specification
│   ├── ARCHITECTURE.md        # Module map, design decisions, data flow
│   ├── WORKFLOW.md            # Operational how-to guides
│   ├── COMPONENT_GUIDE.md     # Component contracts and API reference
│   └── ARCHITECTURE_REVIEW.md # Issue audit and resolution record
│
├── .ai/
│   ├── CLAUDE.md              # Agent instructions
│   ├── AGENT_RULES.md         # Pre-coding rules for AI agents
│   └── CODEX.md               # Code conventions and style guide
│
├── test_data/                 # Bundled demo images (image1.png … image9.png)
├── train_data/                # MNIST cache — git-ignored, created by train.py
│
├── digit_model.keras          # Saved best model — created by train.py (~2 MB)
├── training_curves.png        # Accuracy/loss plot — created by train.py
│
├── Dockerfile                 # Two-stage production image
├── docker-compose.yml         # Convenience compose config
└── requirements.txt           # Pinned Python dependencies
```

---

## Quick Start

### Prerequisites

- Python 3.11+

### Install

```bash
pip install -r requirements.txt
```

### Train

```bash
python src/train.py
```

Downloads MNIST on first run (~12 MB via Keras), trains for up to 30 epochs, and saves `digit_model.keras` + `training_curves.png`.

### Predict

```bash
# Predict on any image file
python src/test.py test_data/image1.png

# Show the preprocessed 28×28 patch in a window
python src/test.py test_data/image1.png --show

# Test a specific MNIST sample by index
python src/test.py --sample 42 --show

# Bulk sanity check (default: 2 000 samples)
python src/test.py
python src/test.py --n 500
```

---

## CLI Reference

### `src/train.py`

```
python src/train.py
```

No arguments. Uses `TrainConfig` defaults from `src/config.py`.

### `src/test.py`

```
python src/test.py [image] [--show] [--model PATH] [--sample N] [--n N]
```

| Argument | Description |
|---|---|
| `image` | Path to a digit image (JPEG, PNG, BMP, TIFF — any resolution) |
| `--show` | Display the preprocessed 28×28 patch in a Matplotlib window |
| `--model PATH` | Override the model path (default: `digit_model.keras`) |
| `--sample N` | Test sample index `N` from local MNIST test data |
| `--n N` | Number of samples for the sanity check (default: 2 000) |

---

## Configuration

All paths and hyperparameters live in `src/config.py`. Override `TrainConfig` fields for experiments:

```python
from config import TrainConfig
cfg = TrainConfig(learning_rate=5e-4, epochs=50, dropout_dense=0.3)
```

| Field | Default | Description |
|---|---|---|
| `learning_rate` | `1e-3` | Initial Adam learning rate |
| `batch_size` | `128` | Mini-batch size |
| `epochs` | `30` | Max training epochs (EarlyStopping may stop earlier) |
| `dropout_conv` | `0.25` | Dropout after each conv block |
| `dropout_dense` | `0.50` | Dropout before the output layer |
| `filters` | `(32, 64, 128)` | Conv filter counts per block |
| `dense_units` | `256` | Units in the fully-connected head |

---

## CNN Architecture

```
Input (28, 28, 1)
│
├── Augmentation: RandomRotation(10%) + RandomZoom(10%) + RandomTranslation(10%)  [train only]
│
├── Conv(32) → BN → Conv(32) → BN → MaxPool → Dropout(0.25)
├── Conv(64) → BN → Conv(64) → BN → MaxPool → Dropout(0.25)
├── Conv(128) → BN → Dropout(0.25)
│
├── GlobalAveragePooling2D
├── Dense(256) → BN → Dropout(0.50)
│
└── Dense(10, softmax)  →  class probabilities (digits 0–9)
```

**Optimizer:** Adam · **Loss:** categorical crossentropy · **Target accuracy:** >99% on MNIST test split

---

## Docker

### Build

```bash
docker build -t mnist-digit .
```

### Train + infer

```bash
# Train (mounts train_data/ for persistence)
docker run --rm -v $(pwd)/train_data:/app/train_data mnist-digit python src/train.py

# Predict on a bundled sample
docker run --rm mnist-digit

# Predict on a custom image
docker run --rm -v $(pwd)/my_images:/data mnist-digit python src/test.py /data/digit.png
```

### docker compose

```bash
docker compose run --rm mnist python src/train.py
docker compose run --rm mnist python src/test.py test_data/image1.png
docker compose run --rm mnist pytest src/tests/ -v
```

---

## Tests

```bash
pytest src/tests/
```

13 tests covering `preprocess_mnist_array`, `preprocess_file_image`, and `load_local_test_data`. No trained model or internet access required.

---

## Documentation

| File | Description |
|---|---|
| [docs/PROJECT_OVERVIEW.md](docs/PROJECT_OVERVIEW.md) | Full project specification, features, and design rationale |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Module map, dependency graph, and design decisions |
| [docs/WORKFLOW.md](docs/WORKFLOW.md) | Step-by-step operational guides |
| [docs/COMPONENT_GUIDE.md](docs/COMPONENT_GUIDE.md) | Component contracts and API reference |
| [.ai/CODEX.md](.ai/CODEX.md) | Code conventions and style guide |

---

## Tech Stack

| Category | Technology |
|---|---|
| Language | Python 3.11 |
| Deep learning | TensorFlow / Keras |
| Numerical computing | NumPy |
| Image processing | Pillow |
| Visualisation | Matplotlib |
| Testing | pytest |
| Containerisation | Docker (two-stage build) |
