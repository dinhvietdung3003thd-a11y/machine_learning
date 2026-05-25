"""Model building utilities for skin cancer binary classification."""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, metrics

from src import config


def build_model(
    input_shape=config.INPUT_SHAPE,
    base_trainable: bool = False,
    dropout_rate_1: float = 0.3,
    dropout_rate_2: float = 0.2,
    dense_units: int = 128,
) -> tf.keras.Model:
    """Build MobileNetV2-based binary classification model."""
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = base_trainable

    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate_1)(x)
    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.Dropout(dropout_rate_2)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="skin_cancer_mobilenetv2")
    return model


def compile_model(model: tf.keras.Model, learning_rate: float = config.LEARNING_RATE) -> tf.keras.Model:
    """Compile model with binary crossentropy and robust metrics."""
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            metrics.BinaryAccuracy(name="accuracy"),
            metrics.Precision(name="precision"),
            metrics.Recall(name="recall"),
            metrics.AUC(name="auc"),
        ],
    )
    return model


def build_and_compile_model(
    input_shape=config.INPUT_SHAPE,
    learning_rate: float = config.LEARNING_RATE,
) -> tf.keras.Model:
    """Convenience helper to build and compile the default model."""
    model = build_model(input_shape=input_shape)
    return compile_model(model, learning_rate=learning_rate)
