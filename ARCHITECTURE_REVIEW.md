# Architecture Review — mnist-digit

**Date:** 2026-06-10  
**Reviewer:** Claude Code (senior engineer review)  
**Branch:** HigherResolution  
**Files reviewed:** `train.py`, `test.py`  
**Status:** All 15 issues resolved — see [Resolution] notes per issue.

---

## Summary

This was a research prototype built incrementally, with data paths that drifted between scripts, global state coupling, no shared config layer, and a preprocessing function that accumulated three input-type paths with divergent behaviour. All structural issues have been resolved. The project now has a single source of truth for paths and hyperparameters (`config.py`), a clean two-function preprocessing contract, proper exception handling, a test suite, and dependency manifest.

---

## Issues

---

### 1. DATA_DIR Path Mismatch — Silent Runtime Break

**Severity:** Critical | **Status:** ✅ Resolved  
**Impact:** The default `python test.py` mode crashed immediately for any new user. `train.py` wrote to `./train_data/` but `test.py` read from `./data/`.

**Root cause:** Paths were hardcoded as module-level string constants in each script independently, with no shared definition.

**Resolution:** Created `config.py` with `DATA_DIR = Path("./train_data")` and `MODEL_PATH`. Both scripts now import from there. Paths cannot drift again because there is only one definition.

```python
# config.py
DATA_DIR   = Path("./train_data")
MODEL_PATH = Path("digit_model.keras")

# train.py and test.py
from config import DATA_DIR, MODEL_PATH
```

---

### 2. `data_augmentation` Captured from Outer Scope

**Severity:** High | **Status:** ✅ Resolved  
**Impact:** `build_model()` silently depended on `data_augmentation` being defined before it in module scope. Moving, reordering, or importing the function in isolation would break silently.

**Root cause:** The augmentation Sequential model was defined at module scope during iterative development and referenced as a free variable inside the function.

**Resolution:** `augmentation` is now an explicit parameter of `build_model()` with a `None` default. The caller in `main()` creates the augmentation and passes it in.

```python
def build_model(cfg: TrainConfig, input_shape=(28, 28, 1), num_classes=10, augmentation=None):
    inputs = keras.Input(shape=input_shape)
    x = augmentation(inputs) if augmentation is not None else inputs
    ...

# In main():
augmentation = keras.Sequential([...], name="augmentation")
model = build_model(cfg, augmentation=augmentation)
```

---

### 3. No `if __name__ == "__main__":` Guard in `train.py`

**Severity:** High | **Status:** ✅ Resolved  
**Impact:** `import train` triggered MNIST download, full CNN training, and disk writes. The module was untestable and unusable as a library.

**Root cause:** The entire pipeline was written as top-level script statements with no encapsulation.

**Resolution:** All pipeline logic is now inside `main()`. The module imports and function definitions are the only top-level statements.

```python
def main():
    cfg = TrainConfig()
    ...  # full pipeline

if __name__ == "__main__":
    main()
```

---

### 4. Redundant Model Reload After Training

**Severity:** Medium | **Status:** ✅ Resolved  
**Impact:** After training, `train.py` loaded the best checkpoint from disk even though `EarlyStopping(restore_best_weights=True)` already restored the best weights into the in-memory model.

**Root cause:** Defensive pattern from an earlier version before `restore_best_weights` was used.

**Resolution:** Removed the `keras.models.load_model` call. Evaluation now uses the in-memory `model` directly.

```python
# Before
best_model = keras.models.load_model("digit_model.keras")
test_loss, test_acc = best_model.evaluate(x_test, y_test_cat, verbose=0)

# After
test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=0)
```

---

### 5. `preprocess_image` Accepts Three Incompatible Input Types

**Severity:** Medium | **Status:** ✅ Resolved  
**Impact:** A single function handled `str/Path`, `PIL.Image`, and `np.ndarray` with three divergent code paths. The numpy path skipped denoising, cropping, padding, and resizing entirely, meaning the sanity check and real-world file inference evaluated the model on fundamentally different preprocessed inputs.

**Root cause:** Input-type handling was added incrementally to one function without establishing separate contracts.

**Resolution:** Split into two functions with explicit contracts:

```python
def preprocess_mnist_array(arr: np.ndarray) -> np.ndarray:
    """Raw 28x28 uint8 MNIST array → (1, 28, 28, 1) float32 tensor."""
    return (arr.astype("float32") / 255.0)[np.newaxis, :, :, np.newaxis]

def preprocess_file_image(image_input, target_size: int = 28) -> np.ndarray:
    """Arbitrary-resolution file path or PIL Image → (1, 28, 28, 1) float32 tensor."""
    # greyscale → invert → denoise → crop → pad → resize → normalise
    ...
```

`predict()` dispatches to the correct function based on `isinstance` check.

---

### 6. `sys.exit` in Library Functions Instead of Raising

**Severity:** Medium | **Status:** ✅ Resolved  
**Impact:** `sys.exit(1)` inside `load_local_test_data()` and `test_single_local_sample()` made them uncheckable in tests and blocked any caller from handling the error gracefully.

**Root cause:** Error handling written for a single call site (the CLI) leaked into library functions.

**Resolution:** Library functions now raise standard exceptions. `main()` catches them and calls `sys.exit()`.

```python
# load_local_test_data — raises instead of exits
raise FileNotFoundError(
    f"Test data not found in '{DATA_DIR}/'. Run 'python train.py' first."
)

# test_single_local_sample — raises instead of exits
raise IndexError(f"Index {index} out of range (max {len(x_test) - 1})")

# main() — handles cleanly at the boundary
except (FileNotFoundError, IndexError) as exc:
    sys.exit(f"Error: {exc}")
```

---

### 7. Normalization Reimplemented in `sanity_check`

**Severity:** Medium | **Status:** ✅ Resolved  
**Impact:** `sanity_check` manually normalised raw numpy arrays with inline arithmetic instead of calling `preprocess_mnist_array`. Two implementations of the same logic could diverge silently.

**Root cause:** `sanity_check` predates the `preprocess_image` function and was never updated to use it.

**Resolution:** `sanity_check` now uses `preprocess_mnist_array` for both normalization and tensor shaping.

```python
# Before
x_sub = x_test[:n_samples].astype("float32") / 255.0
x_sub_4d = np.expand_dims(x_sub, -1)

# After
x_sub_4d = np.stack([preprocess_mnist_array(img)[0] for img in x_test[:n_samples]])
```

---

### 8. Hardcoded Magic Numbers Throughout Preprocessing

**Severity:** Medium | **Status:** ✅ Resolved  
**Impact:** `0.15`, `7`, `127`, `30`, `3`, `4` appeared as unnamed literals in `test.py`. A reader could not tell whether they were principled choices or arbitrary guesses.

**Resolution:** Named module-level constants in `test.py`:

```python
_CROP_THRESHOLD_RATIO = 0.15
_INVERT_THRESHOLD     = 127
_MARGIN_DIVISOR       = 7
_MIN_MARGIN_PX        = 4
_MEDIAN_FILTER_SIZE   = 3
_BAR_WIDTH_CHARS      = 30
```

---

### 9. Mixed `os.path` vs `pathlib` Conventions

**Severity:** Low–Medium | **Status:** ✅ Resolved  
**Impact:** `train.py` used `os.makedirs` and `os.path.join` while `test.py` used `pathlib.Path`. Inconsistent across a two-file project.

**Resolution:** Standardised on `pathlib.Path` throughout both scripts. `os` import removed from `train.py`.

---

### 10. PIL Import Inside a `for` Loop

**Severity:** Low | **Status:** ✅ Resolved  
**Impact:** `from PIL import Image` was placed inside the loop body. Invisible to import scanners and flagged by linters.

**Resolution:** Moved to the top of `train.py` with the other imports.

---

### 11. Training Data Overwritten on Every Run

**Severity:** Low | **Status:** ✅ Resolved  
**Impact:** `.npy` files were unconditionally overwritten on every `python train.py` call, writing 200 MB+ to disk even when files already existed.

**Resolution:** Existence check before writing:

```python
if not train_images_path.exists():
    np.save(train_images_path, x_train)
    ...
else:
    log.info(f"Dataset already cached in '{DATA_DIR}/' — skipping save.")
```

---

### 12. No Dependency Manifest

**Severity:** Low | **Status:** ✅ Resolved  
**Resolution:** Added `requirements.txt`:

```
tensorflow>=2.15
numpy>=1.24
Pillow>=10.0
matplotlib>=3.7
pytest>=7.0
```

---

### 13. No Test Suite

**Severity:** Low | **Status:** ✅ Resolved  
**Impact:** The preprocessing pipeline — the most complex part of the project — had no regression coverage.

**Resolution:** Added `tests/test_preprocess.py` with 13 pytest tests covering `preprocess_mnist_array`, `preprocess_file_image`, and `load_local_test_data`. `conftest.py` at the project root adds the project root to `sys.path`.

```bash
pytest tests/
```

---

### 14. No Logging Framework — Pure `print()`

**Severity:** Low | **Status:** ✅ Resolved  
**Impact:** Verbosity could not be controlled programmatically. No log levels, no routing to file.

**Resolution:** Both scripts now use `logging.basicConfig(level=logging.INFO, format="%(message)s")` and `log = logging.getLogger(__name__)`. All `print()` calls replaced with `log.info()` / `log.error()`.

---

### 15. Training Hyperparameters Hardcoded Without a Config Layer

**Severity:** Low | **Status:** ✅ Resolved  
**Impact:** Learning rate, batch size, epoch count, dropout rates, and filter counts were scattered as inline literals. Running experiments required editing source.

**Resolution:** `TrainConfig` dataclass in `config.py`. `build_model()` and `main()` consume it.

```python
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

## Issue Summary Table

| # | Issue | Severity | Status |
|---|-------|----------|--------|
| 1 | DATA_DIR path mismatch silently breaks sanity check | **Critical** | ✅ Fixed |
| 2 | `data_augmentation` captured from outer scope | **High** | ✅ Fixed |
| 3 | No `if __name__ == "__main__":` guard in train.py | **High** | ✅ Fixed |
| 4 | Redundant model reload from disk after training | Medium | ✅ Fixed |
| 5 | `preprocess_image` three divergent input type paths | Medium | ✅ Fixed |
| 6 | `sys.exit` in library function — uncheckable | Medium | ✅ Fixed |
| 7 | Normalization duplicated in `sanity_check` | Medium | ✅ Fixed |
| 8 | Magic numbers throughout preprocessing | Medium | ✅ Fixed |
| 9 | Mixed `os.path` vs `pathlib` conventions | Low–Med | ✅ Fixed |
| 10 | PIL import inside `for` loop | Low | ✅ Fixed |
| 11 | Training data overwritten on every run | Low | ✅ Fixed |
| 12 | No requirements.txt | Low | ✅ Fixed |
| 13 | No test suite | Low | ✅ Fixed |
| 14 | Pure `print()` — no logging levels | Low | ✅ Fixed |
| 15 | Hyperparameters hardcoded without config layer | Low | ✅ Fixed |
