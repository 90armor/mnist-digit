# Workflow Guide — mnist-digit

## 1. Local Development Setup

```bash
git clone <repo-url>
cd mnist-digit
pip install -r requirements.txt
```

---

## 2. Training Workflow

### Step 1: Run training

```bash
python src/train.py
```

What happens:

1. Downloads MNIST via Keras on first run (~12 MB, cached automatically)
2. Saves raw arrays to `train_data/` (skipped on subsequent runs)
3. Saves 10 PNG previews to `train_data/sample_images/`
4. Normalises and augments data (rotation ±10%, zoom ±10%, translation ±10%)
5. Trains the CNN for up to 30 epochs with EarlyStopping
6. Saves the best checkpoint (by `val_accuracy`) to `digit_model.keras`
7. Saves accuracy/loss curves to `training_curves.png`

Artifacts produced:

```
digit_model.keras          (~2 MB)
training_curves.png
train_data/
  train_images.npy         (60,000 × 28 × 28, uint8)
  train_labels.npy
  test_images.npy          (10,000 × 28 × 28, uint8)
  test_labels.npy
  sample_images/           (10 PNG previews)
```

Expected result: >99% accuracy on the MNIST test split.

### Step 2: Inspect the training curves

```bash
open training_curves.png       # macOS
xdg-open training_curves.png   # Linux
```

---

## 3. Inference Workflow

Requires `digit_model.keras` produced by Step 2.

### Predict on a custom image

Any format (JPEG, PNG, BMP, TIFF), any resolution, any colour mode.

```bash
python src/test.py path/to/digit.png
python src/test.py path/to/digit.png --show   # display 28×28 patch in a window
```

### Test a specific MNIST sample by index

```bash
python src/test.py --sample 0 --show
python src/test.py --sample 42
```

### Bulk sanity check

```bash
python src/test.py              # default: 2000 samples
python src/test.py --n 10000    # all 10,000 MNIST test samples
```

---

## 4. Docker Workflow

### Build

```bash
docker build -t mnist-digit .
```

### Train inside Docker

```bash
docker run --rm \
  -v $(pwd)/train_data:/app/train_data \
  mnist-digit python src/train.py
```

### Predict inside Docker

```bash
docker run --rm \
  -v $(pwd)/test_data:/app/test_data \
  mnist-digit python src/test.py test_data/image1.png
```

### Using docker compose

```bash
# Train
docker compose run --rm mnist python src/train.py

# Predict
docker compose run --rm mnist python src/test.py test_data/image1.png

# Sanity check
docker compose run --rm mnist python src/test.py --n 100

# Run tests
docker compose run --rm mnist pytest src/tests/ -v
```

---

## 5. Testing Workflow

No trained model or internet access required.

```bash
pytest src/tests/           # run all 13 tests
pytest src/tests/ -v        # verbose output
pytest src/tests/ -k "mnist"  # filter by name
```

---

## 6. Hyperparameter Experiment Workflow

Override `TrainConfig` fields in a local script or directly in `src/config.py`:

```python
# experiment.py (run from project root)
import sys
sys.path.insert(0, "src")

from config import TrainConfig
from train import build_model, main

# Modify defaults
import config
config.TrainConfig.learning_rate = 5e-4
config.TrainConfig.epochs = 50
config.TrainConfig.dropout_dense = 0.3

main()
```

Available fields and defaults: see `src/config.py` → `TrainConfig` or `docs/COMPONENT_GUIDE.md`.

---

## 7. Adding a New Hyperparameter

1. Add the field to `TrainConfig` in `src/config.py`
2. Pass `cfg.<field>` where used in `src/train.py` (`build_model` or `main`)
3. Update `docs/COMPONENT_GUIDE.md` TrainConfig field table
4. No changes needed in `src/test.py` — inference is hyperparameter-independent
