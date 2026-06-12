# Codex — Code Conventions

## Python Version

Python 3.11+. No type-stub files. Type hints on public function signatures where they add clarity.

## Naming

| Symbol | Convention | Example |
|--------|-----------|---------|
| Module-level private constant | `_UPPER_SNAKE` | `_CROP_THRESHOLD_RATIO` |
| Public exported constant | `UPPER_SNAKE` | `DATA_DIR`, `MODEL_PATH` |
| Config dataclass | `TitleCase` | `TrainConfig` |
| Functions and methods | `lower_snake` | `build_model`, `preprocess_file_image` |
| Private helper functions | `_lower_snake` | `_load_image` |

## File & Folder Conventions

```
src/         Python source modules
src/tests/   pytest tests
docs/        Markdown documentation
.ai/         Agent instructions and conventions
test_data/   Inference sample images
train_data/  MNIST cache (git-ignored, generated at runtime)
```

## Paths

All paths are defined once in `src/config.py` using `__file__`-relative absolute paths. Never hardcode `Path("./train_data")` or similar in other modules.

```python
# config.py — single source of truth
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR    = _PROJECT_ROOT / "train_data"
MODEL_PATH  = _PROJECT_ROOT / "digit_model.keras"
CURVES_PATH = _PROJECT_ROOT / "training_curves.png"

# train.py / test.py — always import from config
from config import DATA_DIR, MODEL_PATH
```

## Logging

Use `logging` everywhere; no bare `print()` calls.

```python
import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

log.info("...")
log.error("...")
```

## Error Handling

Library functions raise standard exceptions. Only `main()` calls `sys.exit()`.

```python
# Library function — raises
def load_local_test_data():
    if not images_path.exists():
        raise FileNotFoundError("... Run 'python src/train.py' first.")

# Entry point — catches and exits
def main():
    try:
        load_local_test_data()
    except FileNotFoundError as exc:
        sys.exit(f"Error: {exc}")
```

## Imports

Order within each group, separated by blank lines:

1. stdlib (`sys`, `logging`, `argparse`, `pathlib`)
2. third-party (`numpy`, `tensorflow`, `PIL`, `matplotlib`)
3. local (`from config import ...`)

## `if __name__ == "__main__":`

Every entry-point script (`train.py`, `test.py`) must guard execution:

```python
def main():
    ...

if __name__ == "__main__":
    main()
```

This makes the module safe to import in tests.

## Testing

- Framework: pytest
- Location: `src/tests/`
- Run from project root: `pytest src/tests/`
- Tests must not require a trained model or internet access
- Monkeypatch `DATA_DIR` for filesystem-dependent tests
- `src/conftest.py` adds `src/` to `sys.path` so tests can `import config`, `import test`

## Documentation Updates

When adding or modifying a public interface:

1. Update `docs/ARCHITECTURE.md` module map and flow diagrams
2. Update `docs/COMPONENT_GUIDE.md` component contract
3. Update `docs/WORKFLOW.md` if a user-facing procedure changes
