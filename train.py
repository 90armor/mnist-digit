"""
train.py - Train a CNN for digit recognition on MNIST
Saves model as 'digit_model.keras' after training.
Also saves test data locally to ./train_data/ for use by test.py

To Train the model:
pip install tensorflow pillow matplotlib numpy
python train.py
"""

import logging

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from PIL import Image

from config import DATA_DIR, MODEL_PATH, TrainConfig

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

tf.random.set_seed(42)
np.random.seed(42)


def build_model(cfg: TrainConfig, input_shape=(28, 28, 1), num_classes=10, augmentation=None):
    inputs = keras.Input(shape=input_shape)
    x = augmentation(inputs) if augmentation is not None else inputs

    f = cfg.filters
    x = layers.Conv2D(f[0], 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(f[0], 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(cfg.dropout_conv)(x)

    x = layers.Conv2D(f[1], 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(f[1], 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(cfg.dropout_conv)(x)

    x = layers.Conv2D(f[2], 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(cfg.dropout_conv)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(cfg.dense_units, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(cfg.dropout_dense)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs, name="digit_cnn")


def main():
    cfg = TrainConfig()

    # ── 1. Load MNIST & Save Test Data Locally ───────────────────────────────
    DATA_DIR.mkdir(exist_ok=True)

    train_images_path = DATA_DIR / "train_images.npy"
    train_labels_path = DATA_DIR / "train_labels.npy"
    test_images_path  = DATA_DIR / "test_images.npy"
    test_labels_path  = DATA_DIR / "test_labels.npy"

    log.info("Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    if not train_images_path.exists():
        log.info(f"Saving dataset to '{DATA_DIR}/' ...")
        np.save(train_images_path, x_train)
        np.save(train_labels_path, y_train)
        np.save(test_images_path,  x_test)
        np.save(test_labels_path,  y_test)
        log.info(f"  Saved: train_images.npy  {x_train.shape}  -> {x_train.nbytes // 1024 // 1024} MB")
        log.info(f"  Saved: train_labels.npy  {y_train.shape}")
        log.info(f"  Saved: test_images.npy   {x_test.shape}   -> {x_test.nbytes // 1024 // 1024} MB")
        log.info(f"  Saved: test_labels.npy   {y_test.shape}")
    else:
        log.info(f"Dataset already cached in '{DATA_DIR}/' — skipping save.")

    samples_dir = DATA_DIR / "sample_images"
    samples_dir.mkdir(exist_ok=True)
    for i in range(10):
        sample = Image.fromarray(x_test[i], mode="L")
        sample.save(samples_dir / f"digit_{y_test[i]}_sample{i}.png")
    log.info(f"  Saved: 10 sample PNGs  ->  {samples_dir}/")

    # ── 2. Preprocess ────────────────────────────────────────────────────────
    x_train = x_train.astype("float32") / 255.0
    x_test  = x_test .astype("float32") / 255.0
    x_train = np.expand_dims(x_train, -1)
    x_test  = np.expand_dims(x_test,  -1)

    y_train_cat = keras.utils.to_categorical(y_train, 10)
    y_test_cat  = keras.utils.to_categorical(y_test,  10)

    log.info(f"\nTrain: {x_train.shape}  |  Test: {x_test.shape}")

    # ── 3. Data Augmentation ─────────────────────────────────────────────────
    augmentation = keras.Sequential([
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomTranslation(0.1, 0.1),
    ], name="augmentation")

    # ── 4. Build CNN ─────────────────────────────────────────────────────────
    model = build_model(cfg, augmentation=augmentation)
    model.summary()

    # ── 5. Compile ───────────────────────────────────────────────────────────
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=cfg.learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # ── 6. Callbacks ─────────────────────────────────────────────────────────
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            str(MODEL_PATH), save_best_only=True, monitor="val_accuracy", verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=8, restore_best_weights=True, verbose=1,
        ),
    ]

    # ── 7. Train ─────────────────────────────────────────────────────────────
    log.info("\nTraining...")
    history = model.fit(
        x_train, y_train_cat,
        epochs=cfg.epochs, batch_size=cfg.batch_size,
        validation_data=(x_test, y_test_cat),
        callbacks=callbacks, verbose=1,
    )

    # ── 8. Evaluate ──────────────────────────────────────────────────────────
    log.info("\nEvaluating best model on test set...")
    test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=0)
    log.info(f"Test accuracy : {test_acc * 100:.2f}%")
    log.info(f"Test loss     : {test_loss:.4f}")
    log.info(f"\nModel saved  ->  {MODEL_PATH}")

    # ── 9. Plot Training Curves ───────────────────────────────────────────────
    _, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["accuracy"],     label="Train")
    axes[0].plot(history.history["val_accuracy"], label="Val")
    axes[0].set_title("Accuracy"); axes[0].set_xlabel("Epoch"); axes[0].legend()
    axes[1].plot(history.history["loss"],     label="Train")
    axes[1].plot(history.history["val_loss"], label="Val")
    axes[1].set_title("Loss"); axes[1].set_xlabel("Epoch"); axes[1].legend()
    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    log.info("Training curves saved  ->  training_curves.png")
    plt.show()

    log.info("\nProject files after training:")
    log.info(f"  {MODEL_PATH}")
    log.info("  training_curves.png")
    log.info(f"  {DATA_DIR}/")
    log.info("    train_images.npy   (60,000 samples, 28x28)")
    log.info("    train_labels.npy")
    log.info("    test_images.npy    (10,000 samples, 28x28)")
    log.info("    test_labels.npy")
    log.info("    sample_images/     (10 PNG previews)")


if __name__ == "__main__":
    main()
