"""Prediction utilities and CLI for skin cancer binary classification."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError

from src import config

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg"}
DEFAULT_THRESHOLD = 0.5


def load_trained_model() -> tf.keras.Model:
    """Load trained Keras model from config.MODEL_PATH."""
    model_path = Path(config.MODEL_PATH)
    if not model_path.exists() or not model_path.is_file():
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    try:
        return tf.keras.models.load_model(str(model_path))
    except Exception as exc:  # pragma: no cover - defensive for runtime env/model issues
        raise RuntimeError(f"Failed to load model from {model_path}: {exc}") from exc


def validate_image_path(image_path: str | Path) -> Path:
    """Validate that image path exists and has a supported extension."""
    if image_path is None:
        raise ValueError("Image path cannot be None.")

    path = Path(image_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")
    if not path.is_file():
        raise ValueError(f"Provided image path is not a file: {path}")

    extension = path.suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise ValueError(
            f"Unsupported image extension '{extension}'. Supported extensions: {allowed}"
        )

    return path


def preprocess_single_image(image_path: str | Path) -> np.ndarray:
    """Load image and preprocess to match model input pipeline."""
    path = validate_image_path(image_path)

    try:
        with Image.open(path) as image:
            rgb_image = image.convert("RGB")
            resized_image = rgb_image.resize(config.IMAGE_SIZE)
            image_array = np.asarray(resized_image, dtype=np.float32) / 255.0
    except UnidentifiedImageError as exc:
        raise ValueError(f"File is not a valid image or is corrupted: {path}") from exc
    except OSError as exc:
        raise ValueError(f"Failed to read image file: {path}. Error: {exc}") from exc

    if image_array.shape != (
        config.IMAGE_HEIGHT,
        config.IMAGE_WIDTH,
        config.IMAGE_CHANNELS,
    ):
        raise ValueError(
            "Unexpected preprocessed image shape. "
            f"Got {image_array.shape}, expected "
            f"({config.IMAGE_HEIGHT}, {config.IMAGE_WIDTH}, {config.IMAGE_CHANNELS})."
        )

    return np.expand_dims(image_array, axis=0)


def format_prediction_result(
    malignant_probability: float,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, Any]:
    """Convert malignant probability to output dictionary."""
    if not 0.0 <= malignant_probability <= 1.0:
        raise ValueError(
            "Model probability is out of range [0, 1]: "
            f"{malignant_probability}"
        )

    malignant_probability = float(malignant_probability)
    benign_probability = float(1.0 - malignant_probability)

    is_malignant = malignant_probability >= threshold
    label = config.MALIGNANT_LABEL if is_malignant else config.BENIGN_LABEL
    prediction_name = config.CLASS_NAMES[label]
    confidence = malignant_probability if is_malignant else benign_probability

    return {
        "prediction": prediction_name,
        "label": int(label),
        "confidence": float(confidence),
        "malignant_probability": malignant_probability,
        "benign_probability": benign_probability,
    }


def predict_image(
    image_path: str | Path,
    model: tf.keras.Model | None = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, Any]:
    """Run end-to-end prediction for a single image path."""
    inference_model = model or load_trained_model()
    image_batch = preprocess_single_image(image_path)

    probabilities = inference_model.predict(image_batch, verbose=0)
    if probabilities.size == 0:
        raise RuntimeError("Model returned empty predictions.")

    malignant_probability = float(np.asarray(probabilities).reshape(-1)[0])
    return format_prediction_result(malignant_probability, threshold=threshold)


def main() -> int:
    """CLI entry point for image prediction."""
    if len(sys.argv) < 2:
        print("Usage: python -m src.predict <image_path>")
        return 2

    image_path = sys.argv[1]

    try:
        result = predict_image(image_path, threshold=DEFAULT_THRESHOLD)
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1

    print("Prediction completed successfully.")
    print(
        "Result summary: "
        f"{result['prediction']} (label={result['label']}), "
        f"confidence={result['confidence']:.4f}"
    )
    print("Detailed result:")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
