import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms


class OCTADataset(Dataset):
    """
    OCTA-500 Dataset Loader
    Loads:
        - OCTA Image
        - Vessel Label
        - Skeleton Label
    """

    def __init__(self, image_dir, label_dir, skeleton_dir):

        self.image_dir = image_dir
        self.label_dir = label_dir
        self.skeleton_dir = skeleton_dir

        # Read all image filenames
        self.image_files = sorted(
            [f for f in os.listdir(image_dir) if f.endswith(".bmp")]
        )

        # Convert image to PyTorch Tensor
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):

        filename = self.image_files[idx]

        image_path = os.path.join(self.image_dir, filename)
        label_path = os.path.join(self.label_dir, filename)
        skeleton_path = os.path.join(self.skeleton_dir, filename)

        # Read Images
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        skeleton = cv2.imread(skeleton_path, cv2.IMREAD_GRAYSCALE)

        # Normalize (0-255 → 0-1)
        image = image.astype(np.float32) / 255.0
        label = label.astype(np.float32) / 255.0
        skeleton = skeleton.astype(np.float32) / 255.0

        # Convert to Tensor
        image = self.transform(image)
        label = self.transform(label)
        skeleton = self.transform(skeleton)

        return image, label, skeleton
    
   # ==========================================================
# Test Dataset Loader
# ==========================================================

if __name__ == "__main__":

    from config import (
        TRAIN_IMAGE_DIR,
        TRAIN_LABEL_DIR,
        TRAIN_SKELETON_DIR,
    )

    dataset = OCTADataset(
        TRAIN_IMAGE_DIR,
        TRAIN_LABEL_DIR,
        TRAIN_SKELETON_DIR
    )

    print("=" * 40)
    print("Dataset Loaded Successfully")
    print("=" * 40)

    print("Total Images :", len(dataset))

    image, label, skeleton = dataset[0]

    print("Image Shape    :", image.shape)
    print("Label Shape    :", label.shape)
    print("Skeleton Shape :", skeleton.shape)