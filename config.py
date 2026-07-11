import torch
import os

# ==========================================================
# Dataset Paths
# ==========================================================

DATASET_PATH = r"D:\Topology_OCTA\Dataset\OCTA-500"

TRAIN_IMAGE_DIR = os.path.join(
    DATASET_PATH,
    "Training",
    "OCTA500_3M_train_images"
)

TRAIN_LABEL_DIR = os.path.join(
    DATASET_PATH,
    "Training",
    "OCTA500_3M_train_labels"
)

TRAIN_SKELETON_DIR = os.path.join(
    DATASET_PATH,
    "Training",
    "Skeletons"
)

VAL_IMAGE_DIR = os.path.join(
    DATASET_PATH,
    "Validation",
    "OCTA500_3M_val_images"
)

VAL_LABEL_DIR = os.path.join(
    DATASET_PATH,
    "Validation",
    "OCTA500_3M_val_labels"
)

VAL_SKELETON_DIR = os.path.join(
    DATASET_PATH,
    "Validation",
    "Skeletons"
)

TEST_IMAGE_DIR = os.path.join(
    DATASET_PATH,
    "Testing",
    "OCTA500_3M_test_images"
)

TEST_LABEL_DIR = os.path.join(
    DATASET_PATH,
    "Testing",
    "OCTA500_3M_test_labels"
)

TEST_SKELETON_DIR = os.path.join(
    DATASET_PATH,
    "Testing",
    "Skeletons"
)

# ==========================================================
# Image Settings
# ==========================================================

IMAGE_SIZE = 304

# ==========================================================
# Training Hyperparameters
# ==========================================================

BATCH_SIZE = 8
NUM_EPOCHS = 5
LEARNING_RATE = 1e-4

# ==========================================================
# Device
# ==========================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================================
# Checkpoints
# ==========================================================

CHECKPOINT_DIR = "Checkpoints"

# ==========================================================
# Logs
# ==========================================================

LOG_DIR = "logs"