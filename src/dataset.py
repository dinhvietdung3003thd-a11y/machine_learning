"""Utilities for loading and preparing HAM10000 datasets with TensorFlow."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from src import config


# Priority order required by project prompt.
_IMAGE_SEARCH_DIRS = [
    Path(config.BASE_DIR) / "dataset" / "images_part_1",
    Path(config.BASE_DIR) / "dataset" / "images_part_2",
    Path(config.BASE_DIR) / "dataset" / "images_sample",
]


# Metadata path preference: use config first, then fallback to real repo structure.
_METADATA_CANDIDATES = [
    Path(config.METADATA_PATH),
    Path(config.BASE_DIR) / "dataset" / "metadata" / "HAM10000_metadata.csv",
]


def load_metadata() -> pd.DataFrame:
    """Load metadata CSV and validate required columns."""
    metadata_path = next((p for p in _METADATA_CANDIDATES if p.exists()), None)
    if metadata_path is None:
        candidates = " | ".join(str(p) for p in _METADATA_CANDIDATES)
        raise FileNotFoundError(
            f"Metadata file not found. Expected one of: {candidates}"
        )

    df = pd.read_csv(metadata_path)

    required_columns = {"image_id", "dx"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"Metadata is missing required columns: {sorted(missing)}")

    print(f"Loaded metadata: {metadata_path}")
    print(f"Metadata rows: {len(df)}")
    return df


def create_binary_target(df: pd.DataFrame) -> pd.DataFrame:
    """Create binary target column (mel=1, others=0)."""
    data = df.copy()
    data["target"] = (data["dx"] == "mel").astype("int32")
    return data


def resolve_image_path(image_id: str) -> Optional[str]:
    """Resolve image path for an image_id with directory fallback order."""
    filename = f"{image_id}.jpg"

    for directory in _IMAGE_SEARCH_DIRS:
        candidate = directory / filename
        if candidate.exists():
            return str(candidate)

    return None


def add_image_paths(df: pd.DataFrame) -> pd.DataFrame:
    """Add image_path column resolved from image_id."""
    data = df.copy()
    data["image_path"] = data["image_id"].map(resolve_image_path)
    return data


def filter_existing_images(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows with existing images and print summary stats."""
    total_rows = len(df)
    filtered = df[df["image_path"].notna()].copy()
    found_rows = len(filtered)
    missing_rows = total_rows - found_rows

    print(f"Metadata rows total: {total_rows}")
    print(f"Images found: {found_rows}")
    print(f"Images missing: {missing_rows}")

    if found_rows == 0:
        raise ValueError("No images found from metadata. Check dataset folder paths.")

    if found_rows < 50:
        print(
            "[WARNING] Fewer than 50 images were found. "
            "This is suitable for pipeline testing only, not real training."
        )

    return filtered


def _can_stratify(df: pd.DataFrame) -> bool:
    """Return True if stratified splitting is safe for a two-step split."""
    if "target" not in df.columns:
        return False

    value_counts = df["target"].value_counts()
    if len(value_counts) < 2:
        return False

    # conservative threshold to avoid split failures on small datasets
    return bool((value_counts >= 3).all())


def split_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataframe into train/val/test with 70/15/15 ratio."""
    if len(df) < 3:
        raise ValueError("Need at least 3 samples to create train/val/test splits.")

    stratify_main = df["target"] if _can_stratify(df) else None
    if stratify_main is None:
        print("[WARNING] Stratify disabled for first split due to small/imbalanced classes.")

    train_df, temp_df = train_test_split(
        df,
        test_size=1.0 - config.TRAIN_RATIO,
        random_state=config.RANDOM_SEED,
        stratify=stratify_main,
    )

    temp_can_stratify = _can_stratify(temp_df)
    stratify_temp = temp_df["target"] if temp_can_stratify else None
    if stratify_temp is None:
        print("[WARNING] Stratify disabled for second split due to small/imbalanced classes.")

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=config.RANDOM_SEED,
        stratify=stratify_temp,
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def preprocess_image(image_path: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Read, decode, resize, and normalize an image for model input."""
    image_bytes = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image_bytes, channels=3)
    image = tf.image.resize(image, config.IMAGE_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


def build_tf_dataset(df: pd.DataFrame, shuffle: bool = False) -> tf.data.Dataset:
    """Build tf.data.Dataset from dataframe image_path and target columns."""
    paths = df["image_path"].astype(str).tolist()
    labels = df["target"].astype("int32").tolist()

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    if shuffle:
        buffer_size = max(len(df), config.BATCH_SIZE * 4)
        dataset = dataset.shuffle(buffer_size=buffer_size, seed=config.RANDOM_SEED)

    dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(config.BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def get_datasets() -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Run full data pipeline and return train/val/test tf.data datasets."""
    df = load_metadata()
    df = create_binary_target(df)
    df = add_image_paths(df)
    df = filter_existing_images(df)

    train_df, val_df, test_df = split_dataframe(df)

    print(f"Split sizes => train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")
    print("Target distribution (train):")
    print(train_df["target"].value_counts().sort_index())
    print("Target distribution (val):")
    print(val_df["target"].value_counts().sort_index())
    print("Target distribution (test):")
    print(test_df["target"].value_counts().sort_index())

    train_ds = build_tf_dataset(train_df, shuffle=True)
    val_ds = build_tf_dataset(val_df, shuffle=False)
    test_ds = build_tf_dataset(test_df, shuffle=False)

    return train_ds, val_ds, test_ds


if __name__ == "__main__":
    print("Running dataset pipeline check...")
    train_ds, val_ds, test_ds = get_datasets()
    print("TensorFlow datasets built successfully.")
    print(f"Train dataset object: {train_ds}")
    print(f"Validation dataset object: {val_ds}")
    print(f"Test dataset object: {test_ds}")
