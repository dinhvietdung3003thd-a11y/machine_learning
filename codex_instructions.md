# Codex Instructions

This repository is a TensorFlow/Keras + Flask project for skin cancer risk prediction using HAM10000.

## Core constraints

- Keep **TensorFlow/Keras** as ML framework.
- Keep **Flask** as web framework.
- Keep **binary classification** only.
- Keep label mapping:
  - `dx == "mel"` -> malignant (`1`)
  - all other `dx` -> benign (`0`)
- Keep prediction threshold:
  - sigmoid `>= 0.5` -> malignant
  - sigmoid `< 0.5` -> benign

## Dataset layout (current repo)

```text
dataset/
├── metadata/
│   └── HAM10000_metadata.csv
├── images_part_1/
├── images_part_2/
└── images_sample/
```

Do not require physical `train/val/test` folders. Data split must be done in code.

## Data pipeline requirements

- Read metadata from `dataset/metadata/HAM10000_metadata.csv`.
- Resolve image files by `image_id` from:
  1. `dataset/images_part_1/`
  2. `dataset/images_part_2/`
  3. `dataset/images_sample/`
- Preprocess: RGB, resize `224x224`, normalize to `[0,1]`.

## Model requirements

- Transfer learning backbone: MobileNetV2 or EfficientNetB0.
- Final layer: `Dense(1, activation="sigmoid")`.
- Loss: `binary_crossentropy`.
- Metrics: accuracy, precision, recall, AUC.

## Runtime commands

```bash
python -m src.dataset
python -m src.train
python -m src.predict dataset/images_sample/<image_name>.jpg
python app.py
```

## Repository hygiene

- Do not commit full dataset image folders.
- Do not commit trained model binaries.
- Do not commit generated upload/output artifacts.

## Disclaimer

Educational use only. Not a substitute for medical diagnosis.
