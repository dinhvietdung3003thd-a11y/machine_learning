"""Grad-CAM utilities for skin cancer model explainability."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError

from src import config
from src.predict import preprocess_single_image, load_trained_model


def load_model_for_gradcam(model: tf.keras.Model | None = None) -> tf.keras.Model:
    """Return provided model or load trained model from disk."""
    return model or load_trained_model()


def _find_last_conv_layer_recursive(model: tf.keras.Model):
    """Recursively search the last convolution-compatible layer in reverse order."""
    conv_types = (
        tf.keras.layers.Conv2D,
        tf.keras.layers.DepthwiseConv2D,
        tf.keras.layers.SeparableConv2D,
    )

    for layer in reversed(model.layers):
        if isinstance(layer, conv_types):
            return layer

        if isinstance(layer, tf.keras.Model):
            nested_match = _find_last_conv_layer_recursive(layer)
            if nested_match is not None:
                return nested_match

    return None


def find_last_conv_layer(model: tf.keras.Model):
    """Find final convolution layer, including inside nested models."""
    conv_layer = _find_last_conv_layer_recursive(model)
    if conv_layer is None:
        raise ValueError("No convolutional layer found for Grad-CAM generation.")
    return conv_layer


def preprocess_image_for_gradcam(image_path: str | Path) -> np.ndarray:
    """Preprocess image exactly like prediction pipeline."""
    return preprocess_single_image(image_path)


def make_gradcam_heatmap(
    img_array: np.ndarray,
    model: tf.keras.Model,
    last_conv_layer_name: str | None = None,
) -> np.ndarray:
    """Generate normalized Grad-CAM heatmap in [0, 1]."""
    conv_layer = (
        model.get_layer(last_conv_layer_name)
        if last_conv_layer_name
        else find_last_conv_layer(model)
    )

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[conv_layer.output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if predictions.shape.rank is None:
            raise ValueError("Model output rank is unknown; cannot compute Grad-CAM.")

        if predictions.shape.rank == 2:
            class_channel = predictions[:, 0]
        elif predictions.shape.rank == 1:
            class_channel = predictions
        else:
            raise ValueError(
                f"Unsupported model output shape for Grad-CAM: {predictions.shape}"
            )

    grads = tape.gradient(class_channel, conv_outputs)
    if grads is None:
        raise RuntimeError("Failed to compute gradients for Grad-CAM.")

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_outputs[0]

    heatmap = tf.reduce_sum(conv_output * pooled_grads[tf.newaxis, tf.newaxis, :], axis=-1)
    heatmap = tf.maximum(heatmap, 0)

    max_value = tf.reduce_max(heatmap)
    if float(max_value.numpy()) <= 0.0:
        raise RuntimeError("Grad-CAM heatmap is all zeros.")

    heatmap = heatmap / max_value
    return heatmap.numpy()


def _simple_colormap(heatmap_uint8: np.ndarray) -> np.ndarray:
    """Apply a simple blue->yellow->red style map without extra dependencies."""
    x = heatmap_uint8.astype(np.float32) / 255.0

    r = np.clip(1.5 * x - 0.2, 0.0, 1.0)
    g = np.clip(1.5 * (1.0 - np.abs(x - 0.5) * 2.0), 0.0, 1.0)
    b = np.clip(1.2 * (1.0 - x), 0.0, 1.0)

    colored = np.stack([r, g, b], axis=-1)
    return (colored * 255).astype(np.uint8)


def save_gradcam_overlay(image_path: str | Path, heatmap: np.ndarray, output_path: str | Path) -> Path:
    """Overlay heatmap on original image and save to output path."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with Image.open(image_path) as image:
            original = image.convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError(f"Invalid image for Grad-CAM overlay: {image_path}") from exc
    except OSError as exc:
        raise ValueError(f"Failed to open image for Grad-CAM overlay: {image_path}") from exc

    original_arr = np.asarray(original, dtype=np.uint8)

    heatmap_uint8 = np.uint8(np.clip(heatmap, 0.0, 1.0) * 255)
    heatmap_img = Image.fromarray(heatmap_uint8, mode="L").resize(original.size, Image.BILINEAR)
    heatmap_resized = np.asarray(heatmap_img, dtype=np.uint8)

    heatmap_rgb = _simple_colormap(heatmap_resized)

    alpha = 0.4
    overlay = np.clip(
        (1.0 - alpha) * original_arr.astype(np.float32) + alpha * heatmap_rgb.astype(np.float32),
        0,
        255,
    ).astype(np.uint8)

    Image.fromarray(overlay).save(output_path, format="JPEG", quality=95)
    return output_path


def generate_gradcam(image_path: str | Path, model: tf.keras.Model | None = None) -> str:
    """Generate Grad-CAM overlay for an image and return output filename."""
    inference_model = load_model_for_gradcam(model)
    img_array = preprocess_image_for_gradcam(image_path)
    heatmap = make_gradcam_heatmap(img_array, inference_model)

    original_stem = Path(image_path).stem
    safe_stem = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in original_stem)
    output_filename = f"gradcam_{safe_stem}_{uuid4().hex}.jpg"
    output_path = Path(config.OUTPUT_DIR) / output_filename

    save_gradcam_overlay(image_path=image_path, heatmap=heatmap, output_path=output_path)
    return output_filename
