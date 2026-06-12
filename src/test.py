"""
test.py - Predict handwritten digits from images of ANY resolution.

Usage:
    python src/test.py image.png
    python src/test.py image.jpg --show    # show preprocessed patch + result
    python src/test.py                     # sanity check using local train_data/ files
    python src/test.py --sample 3          # test a specific sample index from local data
"""

import sys
import logging
import argparse
import numpy as np
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from PIL import Image, ImageOps, ImageFilter
import matplotlib.pyplot as plt

from config import DATA_DIR, MODEL_PATH

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# ─── Preprocessing constants ──────────────────────────────────────────────────
_CROP_THRESHOLD_RATIO = 0.15
_INVERT_THRESHOLD     = 127
_MARGIN_DIVISOR       = 7
_MIN_MARGIN_PX        = 4
_MEDIAN_FILTER_SIZE   = 3
_BAR_WIDTH_CHARS      = 30


# ─────────────────────────────────────────────────────────────────────────────
#  Load local test data (saved by train.py)
# ─────────────────────────────────────────────────────────────────────────────

def load_local_test_data():
    """Load test images & labels from DATA_DIR saved by train.py."""
    images_path = DATA_DIR / "test_images.npy"
    labels_path = DATA_DIR / "test_labels.npy"

    if not images_path.exists() or not labels_path.exists():
        raise FileNotFoundError(
            f"Test data not found in '{DATA_DIR}/'. Run 'python train.py' first."
        )

    x_test = np.load(images_path)
    y_test = np.load(labels_path)
    log.info(f"Loaded local test data  ->  {x_test.shape}  from  {DATA_DIR}/")
    return x_test, y_test


# ─────────────────────────────────────────────────────────────────────────────
#  Preprocessing — two explicit paths with clear contracts
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_mnist_array(arr: np.ndarray) -> np.ndarray:
    """Normalise a raw 28x28 MNIST uint8 array to a (1, 28, 28, 1) float32 tensor."""
    return (arr.astype("float32") / 255.0)[np.newaxis, :, :, np.newaxis]


def preprocess_file_image(image_input, target_size: int = 28) -> np.ndarray:
    """
    Full preprocessing pipeline for arbitrary-resolution images.

    Accepts a file path (str or Path) or a PIL Image.
    Returns a normalised (1, target_size, target_size, 1) float32 tensor.
    """
    if isinstance(image_input, (str, Path)):
        img = Image.open(image_input)
    else:
        img = image_input

    img = img.convert("L")

    # Auto-invert: MNIST is white-digit / black-background
    if np.array(img).mean() > _INVERT_THRESHOLD:
        img = ImageOps.invert(img)

    img = img.filter(ImageFilter.MedianFilter(size=_MEDIAN_FILTER_SIZE))

    # Tight crop to bounding box of significant pixels
    arr = np.array(img)
    threshold = arr.max() * _CROP_THRESHOLD_RATIO
    rows = np.any(arr > threshold, axis=1)
    cols = np.any(arr > threshold, axis=0)
    if rows.any() and cols.any():
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        img = img.crop((cmin, rmin, cmax + 1, rmax + 1))

    # Pad to square with proportional margin
    w, h   = img.size
    side   = max(w, h)
    margin = max(_MIN_MARGIN_PX, side // _MARGIN_DIVISOR)
    side  += margin * 2
    padded = Image.new("L", (side, side), 0)
    padded.paste(img, (margin + (side - margin*2 - w)//2,
                       margin + (side - margin*2 - h)//2))

    resized = padded.resize((target_size, target_size), Image.LANCZOS)

    out = np.array(resized).astype("float32") / 255.0
    out = np.clip((out - out.min()) / (out.max() - out.min() + 1e-8), 0, 1)

    return out[np.newaxis, :, :, np.newaxis]


# ─────────────────────────────────────────────────────────────────────────────
#  Prediction
# ─────────────────────────────────────────────────────────────────────────────

def predict(model, image_input, show: bool = False, true_label=None) -> int:
    if isinstance(image_input, np.ndarray):
        tensor = preprocess_mnist_array(image_input)
    else:
        tensor = preprocess_file_image(image_input)

    probs  = model.predict(tensor, verbose=0)[0]
    digit  = int(np.argmax(probs))
    conf   = float(probs[digit]) * 100

    prediction_line = f"\n  Predicted digit : {digit}"
    if true_label is not None:
        status = "CORRECT" if digit == true_label else f"WRONG (true={true_label})"
        prediction_line += f"   [{status}]"
    log.info(prediction_line)
    log.info(f"  Confidence      : {conf:.1f}%")
    log.info("\n  All class probabilities:")
    for i, p in enumerate(probs):
        bar    = "█" * int(p * _BAR_WIDTH_CHARS)
        marker = " <--" if i == digit else ""
        log.info(f"    {i}: {bar:<{_BAR_WIDTH_CHARS}} {p*100:5.1f}%{marker}")

    if show:
        arr_28 = tensor[0, :, :, 0]
        title  = f"Predicted: {digit}  ({conf:.1f}%)"
        if true_label is not None:
            title += f"  |  True: {true_label}"
        plt.figure(figsize=(4, 4))
        plt.imshow(arr_28, cmap="gray")
        plt.title(title, fontsize=13)
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    return digit


# ─────────────────────────────────────────────────────────────────────────────
#  Sanity check using LOCAL train_data/ files
# ─────────────────────────────────────────────────────────────────────────────

def sanity_check(model, n_samples: int = 2000):
    """Evaluate on n_samples from the LOCAL test data saved by train.py."""
    x_test, y_test = load_local_test_data()

    n_samples = min(n_samples, len(x_test))
    y_sub     = y_test[:n_samples]

    x_sub_4d = np.stack([preprocess_mnist_array(img)[0] for img in x_test[:n_samples]])
    preds    = np.argmax(model.predict(x_sub_4d, verbose=1), axis=1)
    accuracy = np.mean(preds == y_sub) * 100

    log.info(f"\nAccuracy on {n_samples} local test samples : {accuracy:.2f}%")

    _, axes = plt.subplots(2, 5, figsize=(12, 5))
    indices = np.random.choice(n_samples, 10, replace=False)
    for ax, idx in zip(axes.flat, indices):
        ax.imshow(x_sub_4d[idx, :, :, 0], cmap="gray")
        color = "green" if preds[idx] == y_sub[idx] else "red"
        ax.set_title(f"Pred:{preds[idx]}  True:{y_sub[idx]}", color=color, fontsize=10)
        ax.axis("off")
    plt.suptitle(f"Sample predictions from local data  |  Accuracy: {accuracy:.2f}%")
    plt.tight_layout()
    plt.show()


def test_single_local_sample(model, index: int, show: bool = True):
    """Test a specific sample index from the local train_data/test_images.npy."""
    x_test, y_test = load_local_test_data()

    if index >= len(x_test):
        raise IndexError(f"Index {index} out of range (max {len(x_test) - 1})")

    log.info(f"\nTesting local sample index: {index}")
    predict(model, x_test[index], show=show, true_label=int(y_test[index]))


# ─────────────────────────────────────────────────────────────────────────────
#  CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Digit predictor – any resolution")
    parser.add_argument("image",   nargs="?", default=None,
                        help="Path to digit image (jpg, png, bmp, tiff …)")
    parser.add_argument("--show",  action="store_true",
                        help="Display the preprocessed 28x28 patch + result")
    parser.add_argument("--model", default=MODEL_PATH,
                        help=f"Path to saved model (default: {MODEL_PATH})")
    parser.add_argument("--sample", type=int, default=None,
                        help="Test a specific index from local train_data/test_images.npy")
    parser.add_argument("--n",     type=int, default=2000,
                        help="Number of local samples to use in sanity check (default: 2000)")
    args = parser.parse_args()

    if not Path(args.model).exists():
        log.error(f"Model file '{args.model}' not found. Run 'python train.py' first.")
        sys.exit(1)

    log.info(f"Loading model  ->  {args.model}")
    model = keras.models.load_model(args.model)

    if args.image:
        img_path = Path(args.image)
        if not img_path.exists():
            log.error(f"Image file '{img_path}' not found.")
            sys.exit(1)
        img = Image.open(img_path)
        log.info(f"Image          ->  {img_path}  ({img_path.stat().st_size // 1024} KB)")
        log.info(f"Original size  ->  {img.size[0]}x{img.size[1]} px  |  mode: {img.mode}")
        predict(model, img_path, show=args.show)

    elif args.sample is not None:
        try:
            test_single_local_sample(model, args.sample, show=args.show)
        except (FileNotFoundError, IndexError) as exc:
            sys.exit(f"Error: {exc}")

    else:
        log.info(f"No image provided – running sanity check on local {DATA_DIR}/ files ...")
        try:
            sanity_check(model, n_samples=args.n)
        except FileNotFoundError as exc:
            sys.exit(f"Error: {exc}")


if __name__ == "__main__":
    main()
