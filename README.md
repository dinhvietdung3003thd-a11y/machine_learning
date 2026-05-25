# Skin Cancer Detection using TensorFlow

A Machine Learning project for skin cancer classification using the HAM10000 dataset, TensorFlow, and Flask.

The system allows users to upload a skin lesion image through a web interface and receive a prediction indicating whether the lesion is likely benign or malignant.

---

# Project Overview

Skin cancer is one of the most common forms of cancer worldwide. Early detection significantly improves treatment outcomes.

This project aims to build an end-to-end machine learning system capable of:

- Training a deep learning model using the HAM10000 dataset
- Classifying skin lesion images
- Predicting whether a lesion is benign or malignant
- Providing a simple web-based interface for image upload and prediction
- Visualizing model attention using Grad-CAM (future development)

This project is developed for educational and research purposes only.

---

# Main Features

Current Features:

- TensorFlow-based image classification
- HAM10000 dataset support
- Binary classification:
  - Benign
  - Malignant
- Image upload through Flask web interface
- Prediction confidence score
- Modular code structure
- Support for future model improvements

Planned Features:

- Grad-CAM heatmap visualization
- Prediction history
- Dashboard analytics
- REST API endpoint
- Model comparison
- Improved user interface
- Cloud deployment

---

# Technology Stack

Backend:

- Python
- Flask

Machine Learning:

- TensorFlow
- Keras
- NumPy
- Pandas
- Scikit-learn

Image Processing:

- Pillow
- OpenCV

Frontend:

- HTML
- CSS
- JavaScript

Visualization:

- Matplotlib
- Grad-CAM (planned)

---

# Dataset

This project uses the HAM10000 dataset.

Dataset name:

Human Against Machine with 10000 training images

The dataset contains dermatoscopic images of pigmented skin lesions.

Main files:

```text
HAM10000_metadata.csv
HAM10000_images_part_1/
HAM10000_images_part_2/