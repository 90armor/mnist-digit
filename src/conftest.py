import sys
from pathlib import Path

# Allow test files to import from the project root (train.py, test.py, config.py).
sys.path.insert(0, str(Path(__file__).parent))
