# Skin Cancer Detection using TensorFlow/Keras

A Machine Learning project for **binary skin cancer risk classification** using the HAM10000 dataset, TensorFlow/Keras, and Flask.

The system allows users to upload a skin lesion image and receive a prediction:

- `0 = benign`
- `1 = malignant`

> Educational purpose only. This project does **not** replace medical diagnosis.

---

## Project Scope

Core components implemented in this repository:

- TensorFlow/Keras training pipeline
- HAM10000 metadata + image-file based loading
- Binary classification (`mel` vs others)
- Flask web interface for upload + prediction
- Grad-CAM visualization (best-effort, non-blocking)

---

## Classification Rules

### Label mapping

From HAM10000 `dx` column:

- `dx == "mel"` → `malignant` (`1`)
- all other `dx` values → `benign` (`0`)

### Prediction threshold

Model output is sigmoid probability of malignant class:

- probability `>= 0.5` → `malignant`
- probability `< 0.5` → `benign`

---

## Repository Structure

```text
.
├── app.py
├── README.md
├── codex_instructions.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── predict.py
│   └── gradcam.py
├── templates/
│   └── index.html
├── static/
│   ├── css/style.css
│   └── js/main.js
├── dataset/
│   ├── metadata/HAM10000_metadata.csv
│   ├── images_part_1/
│   ├── images_part_2/
│   └── images_sample/
├── models/
├── uploads/
└── outputs/
```

---

## Dataset Layout (Expected)

- `dataset/metadata/HAM10000_metadata.csv`
- `dataset/images_part_1/`
- `dataset/images_part_2/`
- `dataset/images_sample/` (small sample set for quick testing)

The pipeline resolves images by `image_id` across `images_part_1`, then `images_part_2`, then `images_sample`.

No physical `train/val/test` folders are required. Splits are created in code.

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Run Commands

### 1) Dataset pipeline check

```bash
python -m src.dataset
```

### 2) Train model

```bash
python -m src.train
```

Model is saved to:

- `models/skin_cancer_model.keras`

### 3) Predict from CLI

```bash
python -m src.predict dataset/images_sample/ISIC_0024306.jpg
```

### 4) Run Flask app

```bash
python app.py
```

Open browser at:

- `http://127.0.0.1:5000`

---

## Web App Behavior

- Accepts: `jpg`, `jpeg`, `png`
- Stores uploads in `uploads/`
- Runs prediction from trained model
- Attempts Grad-CAM generation into `outputs/`
- If Grad-CAM fails, prediction result is still shown
- If trained model is missing, app shows a friendly message

---

## Important Notes

- Keep TensorFlow/Keras and Flask framework as core stack.
- Keep binary classification logic and label mapping unchanged.
- Do not commit full dataset images, trained model files, uploads, or generated outputs.

---

## Disclaimer

This project is for education and experimentation only. It is **not** a medical device and must not be used as a substitute for professional clinical diagnosis.
