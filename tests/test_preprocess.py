"""
Unit tests for the preprocessing pipeline in test.py.

Run with:  pytest tests/
"""

import numpy as np
import pytest
from PIL import Image

# conftest.py at the project root adds the project root to sys.path.
import test as inference


# ─── preprocess_mnist_array ───────────────────────────────────────────────────

def test_mnist_array_output_shape():
    arr = np.full((28, 28), 128, dtype=np.uint8)
    result = inference.preprocess_mnist_array(arr)
    assert result.shape == (1, 28, 28, 1)


def test_mnist_array_output_dtype():
    arr = np.zeros((28, 28), dtype=np.uint8)
    result = inference.preprocess_mnist_array(arr)
    assert result.dtype == np.float32


def test_mnist_array_normalises_to_unit_range():
    arr = np.full((28, 28), 255, dtype=np.uint8)
    result = inference.preprocess_mnist_array(arr)
    assert abs(result.mean() - 1.0) < 1e-5


def test_mnist_array_zero_input_stays_zero():
    arr = np.zeros((28, 28), dtype=np.uint8)
    result = inference.preprocess_mnist_array(arr)
    assert result.max() == 0.0


def test_mnist_array_preserves_relative_ordering():
    bright = np.full((28, 28), 200, dtype=np.uint8)
    dark   = np.full((28, 28), 50,  dtype=np.uint8)
    assert inference.preprocess_mnist_array(bright).mean() > \
           inference.preprocess_mnist_array(dark).mean()


# ─── preprocess_file_image ────────────────────────────────────────────────────

def test_file_image_output_shape_from_pil():
    arr = np.full((50, 50), 30, dtype=np.uint8)
    img = Image.fromarray(arr, mode="L")
    result = inference.preprocess_file_image(img)
    assert result.shape == (1, 28, 28, 1)


def test_file_image_output_dtype():
    img = Image.fromarray(np.full((40, 40), 30, dtype=np.uint8), mode="L")
    result = inference.preprocess_file_image(img)
    assert result.dtype == np.float32


def test_file_image_output_range():
    img = Image.fromarray(np.full((40, 40), 30, dtype=np.uint8), mode="L")
    result = inference.preprocess_file_image(img)
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_file_image_accepts_rgb():
    arr = np.full((50, 50, 3), 30, dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    result = inference.preprocess_file_image(img)
    assert result.shape == (1, 28, 28, 1)


def test_file_image_accepts_file_path(tmp_path):
    arr = np.full((60, 60), 30, dtype=np.uint8)
    p = tmp_path / "digit.png"
    Image.fromarray(arr, mode="L").save(str(p))
    result = inference.preprocess_file_image(p)
    assert result.shape == (1, 28, 28, 1)


def test_file_image_accepts_str_path(tmp_path):
    arr = np.full((60, 60), 30, dtype=np.uint8)
    p = tmp_path / "digit.png"
    Image.fromarray(arr, mode="L").save(str(p))
    result = inference.preprocess_file_image(str(p))
    assert result.shape == (1, 28, 28, 1)


def test_file_image_custom_target_size():
    img = Image.fromarray(np.full((80, 80), 30, dtype=np.uint8), mode="L")
    result = inference.preprocess_file_image(img, target_size=32)
    assert result.shape == (1, 32, 32, 1)


# ─── load_local_test_data ─────────────────────────────────────────────────────

def test_load_local_test_data_raises_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(inference, "DATA_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="Run 'python train.py' first"):
        inference.load_local_test_data()
