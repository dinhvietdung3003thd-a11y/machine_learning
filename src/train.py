"""Training and evaluation script for skin cancer classification."""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from src import config
from src.dataset import get_datasets
from src.model import build_and_compile_model


def get_callbacks(model_path=config.MODEL_PATH) -> list[tf.keras.callbacks.Callback]:
    """Create standard training callbacks monitored by validation AUC."""
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc",
            mode="max",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
    ]


def train(
    model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    epochs: int = config.EPOCHS,
    callbacks: list[tf.keras.callbacks.Callback] | None = None,
) -> tf.keras.callbacks.History:
    """Train model and return history."""
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks or [],
        verbose=1,
    )
    return history


def evaluate_model(model: tf.keras.Model, test_ds: tf.data.Dataset) -> dict[str, float]:
    """Evaluate model on test set and print metric summary."""
    results = model.evaluate(test_ds, verbose=1)
    metrics_dict = dict(zip(model.metrics_names, results))
    print("\nTest metrics:")
    for k, v in metrics_dict.items():
        print(f"- {k}: {v:.6f}")
    return metrics_dict


def _collect_true_labels(test_ds: tf.data.Dataset) -> np.ndarray:
    """Collect ground-truth labels from a batched tf.data dataset."""
    y_true_batches = []
    for _, labels in test_ds:
        y_true_batches.append(labels.numpy())

    if not y_true_batches:
        return np.array([], dtype=np.int32)

    return np.concatenate(y_true_batches).astype(np.int32)


def predict_labels(
    model: tf.keras.Model,
    test_ds: tf.data.Dataset,
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict probabilities and convert to binary labels."""
    y_true = _collect_true_labels(test_ds)
    if y_true.size == 0:
        return y_true, np.array([], dtype=np.float32), np.array([], dtype=np.int32)

    y_prob = model.predict(test_ds, verbose=1).reshape(-1)
    y_pred = (y_prob >= threshold).astype(np.int32)
    return y_true, y_prob, y_pred


def print_classification_outputs(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Print confusion matrix and classification report safely for edge cases."""
    if y_true.size == 0:
        print("\nNo test samples available. Skipping confusion matrix and classification report.")
        return

    print("\nConfusion Matrix (labels: 0=benign, 1=malignant):")
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    print(cm)

    present_classes = np.unique(y_true)
    if present_classes.size < 2:
        only_class = int(present_classes[0])
        print(
            "\n[WARNING] Test set contains only one class "
            f"({only_class}). Classification report for both classes may be uninformative."
        )

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["benign", "malignant"],
        zero_division=0,
    )
    print("\nClassification Report:")
    print(report)


def main() -> None:
    """Main training pipeline."""
    tf.random.set_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    print("Loading datasets...")
    train_ds, val_ds, test_ds = get_datasets()

    print("Building model...")
    model = build_and_compile_model(
        input_shape=config.INPUT_SHAPE,
        learning_rate=config.LEARNING_RATE,
    )
    model.summary()

    callbacks = get_callbacks(config.MODEL_PATH)

    print("Starting training...")
    train(
        model=model,
        train_ds=train_ds,
        val_ds=val_ds,
        epochs=config.EPOCHS,
        callbacks=callbacks,
    )

    print("\nEvaluating model...")
    evaluate_model(model, test_ds)

    y_true, _, y_pred = predict_labels(model, test_ds, threshold=0.5)
    print_classification_outputs(y_true, y_pred)


if __name__ == "__main__":
    main()
