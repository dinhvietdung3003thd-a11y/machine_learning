# Codex Instructions

This repository contains a TensorFlow-based Machine Learning project for skin cancer detection using the HAM10000 dataset.

The goal of this repository is to build a complete AI-powered web application that allows users to upload a skin lesion image and receive a prediction indicating whether the lesion is likely benign or malignant.

---

# Project Goals

The primary objective of this project is to:

1. Train a TensorFlow image classification model using the HAM10000 dataset.
2. Predict whether a skin lesion is benign or malignant.
3. Build a Flask-based web application.
4. Allow users to upload skin images.
5. Display prediction results and confidence scores.
6. Visualize model attention using Grad-CAM.
7. Keep the codebase modular and easy to understand.
8. Maintain compatibility with future improvements.

This project is intended for educational and research purposes.

---

# Core Technology Requirements

The machine learning framework MUST remain:

TensorFlow / Keras

Do not replace TensorFlow with:

- PyTorch
- Scikit-Learn-only models
- XGBoost
- LightGBM
- Other frameworks

TensorFlow is the core technology of this repository.

---

# Dataset

The project uses the HAM10000 dataset.

Dataset structure:

data/
│
├── metadata/
│   └── HAM10000_metadata.csv
│
├── images/
│   ├── ISIC_0000001.jpg
│   ├── ISIC_0000002.jpg
│   └── ...
│
└── images_sample/
    ├── sample_01.jpg
    ├── sample_02.jpg
    └── ...

The complete dataset may not be committed to GitHub because of size limitations.

The folder:

data/images_sample/

contains a small number of example images only.

---

# Classification Strategy

Original HAM10000 labels:

- akiec
- bcc
- bkl
- df
- mel
- nv
- vasc

Convert the problem into binary classification:

mel      -> malignant
others   -> benign

Target rule:

target = 1 if dx == "mel"
target = 0 otherwise

Final output:

0 = benign
1 = malignant

This mapping must be used consistently throughout:

- training
- validation
- testing
- prediction
- evaluation

---

# Data Loading

Data should be loaded from:

HAM10000_metadata.csv

and mapped to image files using:

image_id

Example:

image_id = ISIC_0024306

corresponds to:

data/images/ISIC_0024306.jpg

Avoid using CSV pixel datasets as the primary source.

Preferred approach:

HAM10000_metadata.csv
+
original image files

Reason:

- higher image quality
- better model performance
- realistic inference workflow
- compatibility with transfer learning

---

# Data Splitting

Do NOT require physical folders such as:

train/
validation/
test/

Split data in code using:

train_test_split()

Recommended split:

Train: 70%
Validation: 15%
Test: 15%

Use stratified sampling.

Example:

train_df, temp_df = train_test_split(
    df,
    test_size=0.3,
    stratify=df["target"],
    random_state=42
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    stratify=temp_df["target"],
    random_state=42
)

Advantages:

- no duplicated images
- easier maintenance
- reproducibility
- cleaner repository

---

# Image Preprocessing

All images should be:

1. Loaded as RGB
2. Resized to 224x224
3. Normalized to [0,1]

Expected input shape:

224 x 224 x 3

Example:

image = image / 255.0

---

# Recommended Model

Preferred architecture:

EfficientNetB0

Alternative:

MobileNetV2

Use TensorFlow implementation:

tf.keras.applications.EfficientNetB0

or

tf.keras.applications.MobileNetV2

Transfer learning is recommended.

The classifier head should contain:

Dense(1, activation="sigmoid")

Loss:

binary_crossentropy

Optimizer:

Adam

Metrics:

- accuracy
- precision
- recall
- AUC

---

# Evaluation

Model evaluation should include:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC
- Confusion Matrix

Evaluation must be performed on the test set only.

---

# Repository Structure

Expected structure:

skin-cancer-detection/
│
├── data/
│   ├── metadata/
│   ├── images_sample/
│   └── README.md
│
├── models/
│   └── .gitkeep
│
├── uploads/
│   └── .gitkeep
│
├── outputs/
│   └── .gitkeep
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── predict.py
│   └── gradcam.py
│
├── templates/
│   └── index.html
│
├── static/
│   ├── css/
│   │   └── style.css
│   │
│   └── js/
│       └── main.js
│
├── app.py
├── requirements.txt
├── README.md
├── codex_instructions.md
└── .gitignore

---

# File Responsibilities

src/config.py

Responsible for:

- paths
- constants
- image size
- batch size
- epochs
- model path

Examples:

IMAGE_SIZE
BATCH_SIZE
EPOCHS
MODEL_PATH
IMAGE_DIR

---

src/dataset.py

Responsible for:

- loading metadata
- creating target labels
- creating image paths
- validating image existence
- splitting dataset
- building TensorFlow datasets

Functions should be reusable.

---

src/model.py

Responsible for:

- building neural networks
- transfer learning architecture
- model compilation

This file should not contain training logic.

---

src/train.py

Responsible for:

- dataset loading
- model creation
- training
- validation
- evaluation
- saving model

Output model:

models/skin_cancer_model.keras

or

models/skin_cancer_model.h5

---

src/predict.py

Responsible for:

- loading trained model
- preprocessing uploaded images
- performing prediction
- returning probability
- returning confidence score

Expected return example:

{
    "prediction": "malignant",
    "confidence": 0.92
}

---

src/gradcam.py

Responsible for:

- Grad-CAM generation
- heatmap creation
- overlay generation
- output image saving

Output example:

outputs/gradcam_result.jpg

---

app.py

Responsible for:

- Flask application
- upload endpoint
- prediction endpoint
- result rendering

Should not contain model training logic.

---

# Web Application Requirements

The web application should allow users to:

1. Upload image
2. Preview image
3. Run prediction
4. Display classification result
5. Display confidence score
6. Display malignant probability
7. Display warning message
8. Optionally show Grad-CAM visualization

Expected workflow:

Upload Image
↓
Preprocessing
↓
TensorFlow Model
↓
Prediction
↓
Result

---

# User Interface

The interface should be simple and responsive.

Recommended sections:

- Header
- Upload form
- Image preview
- Prediction card
- Probability bar
- Grad-CAM visualization
- Footer disclaimer

Avoid unnecessary complexity.

The UI should prioritize clarity.

---

# Medical Disclaimer

Every prediction page should contain:

English:

This result is for educational purposes only and is not a medical diagnosis.

Vietnamese:

Kết quả chỉ phục vụ mục đích học tập và không thay thế chẩn đoán y khoa.

This disclaimer must always remain visible.

---

# Future Development

Recommended future improvements:

1. Grad-CAM visualization
2. Prediction history
3. SQLite database
4. Dashboard analytics
5. REST API
6. Docker support
7. User authentication
8. Model version management
9. EfficientNet fine-tuning
10. Multi-class classification

---

# GitHub Guidelines

Do not commit:

data/images/
uploads/
outputs/
venv/
__pycache__/

Large model files should also be excluded:

*.h5
*.keras

Only commit:

- source code
- documentation
- metadata CSV
- sample images
- configuration files

---

# Coding Style

Keep the code:

- modular
- readable
- maintainable
- beginner friendly
- well documented

Prefer simple solutions over complex abstractions.

Avoid over-engineering.

Each module should have a single responsibility.

---

# Development Priority

When extending this repository, prioritize:

1. dataset pipeline
2. model training
3. model evaluation
4. prediction pipeline
5. Flask interface
6. Grad-CAM visualization
7. UI improvements
8. deployment

TensorFlow-based machine learning functionality must always remain the primary focus of the project.