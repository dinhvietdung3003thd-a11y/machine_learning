"""
Global configuration for Skin Cancer Detection Project
"""

from pathlib import Path

# =====================================================
# PROJECT ROOT
# =====================================================

BASE_DIR = Path(__file__).resolve().parent.parent

# =====================================================
# DATASET PATHS
# =====================================================

DATA_DIR = BASE_DIR / "dataset"

METADATA_DIR = DATA_DIR / "metadata"

IMAGE_DIR = DATA_DIR / "images"

SAMPLE_IMAGE_DIR = DATA_DIR / "images_sample"

METADATA_PATH = METADATA_DIR / "HAM10000_metadata.csv"

# =====================================================
# MODEL PATHS
# =====================================================

MODELS_DIR = BASE_DIR / "models"

MODEL_NAME = "skin_cancer_model.keras"

MODEL_PATH = MODELS_DIR / MODEL_NAME

# =====================================================
# UPLOADS
# =====================================================

UPLOAD_DIR = BASE_DIR / "uploads"

# =====================================================
# OUTPUTS
# =====================================================

OUTPUT_DIR = BASE_DIR / "outputs"

# =====================================================
# IMAGE SETTINGS
# =====================================================

IMAGE_WIDTH = 224

IMAGE_HEIGHT = 224

IMAGE_CHANNELS = 3

IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)

INPUT_SHAPE = (
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
    IMAGE_CHANNELS
)

# =====================================================
# DATA SPLIT
# =====================================================

TRAIN_RATIO = 0.70

VALID_RATIO = 0.15

TEST_RATIO = 0.15

# =====================================================
# TRAINING
# =====================================================

BATCH_SIZE = 32

EPOCHS = 20

LEARNING_RATE = 1e-4

RANDOM_SEED = 42

# =====================================================
# LABELS
# =====================================================

BENIGN_LABEL = 0

MALIGNANT_LABEL = 1

CLASS_NAMES = [
    "benign",
    "malignant"
]

# =====================================================
# TARGET MAPPING
# =====================================================

MALIGNANT_DX = [
    "mel"
]

# =====================================================
# FLASK
# =====================================================

HOST = "0.0.0.0"

PORT = 5000

DEBUG = str(__import__("os").environ.get("FLASK_DEBUG", "0")).strip().lower() in {"1", "true", "yes", "on"}

# =====================================================
# GRADCAM
# =====================================================

GRADCAM_OUTPUT_NAME = "gradcam_result.jpg"

# =====================================================
# HELPER
# =====================================================

def ensure_directories():
    """
    Create required directories automatically.
    """

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    UPLOAD_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


# Create folders automatically
ensure_directories()