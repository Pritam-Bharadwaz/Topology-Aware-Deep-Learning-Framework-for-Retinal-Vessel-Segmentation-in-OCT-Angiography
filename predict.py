"""
==========================================================
Prediction Script
==========================================================
"""

import os
import cv2
import torch
import numpy as np

from tqdm import tqdm

from model import SkeletonAwareUNet

from config import (
    TEST_IMAGE_DIR,
    DEVICE,
    IMAGE_SIZE
)


# ==========================================================
# Load Model
# ==========================================================

model = SkeletonAwareUNet().to(DEVICE)

checkpoint = torch.load(
    "Checkpoints/best_model.pth",
    map_location=DEVICE
)

model.load_state_dict(checkpoint["model_state_dict"])

model.eval()

print("Model Loaded Successfully")



# ==========================================================
# Test Images
# ==========================================================

output_dir = "Dataset/OCTA-500/Output"

os.makedirs(output_dir, exist_ok=True)

test_images = sorted(os.listdir(TEST_IMAGE_DIR))

print(f"Total Test Images : {len(test_images)}")


# ==========================================================
# Prediction
# ==========================================================

with torch.no_grad():

    for image_name in tqdm(test_images):

        # Read image
        image_path = os.path.join(TEST_IMAGE_DIR, image_name)

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            print(f"Could not read {image_name}")
            continue

        # Resize
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))

        # Normalize
        image = image.astype(np.float32) / 255.0

        # Convert to Tensor
        image = torch.from_numpy(image)

        image = image.unsqueeze(0).unsqueeze(0)

        image = image.to(DEVICE)

        # Prediction
        prediction = model(image)

        prediction = prediction.squeeze().cpu().numpy()

        # Convert probability map to binary mask
        prediction = (prediction > 0.5).astype(np.uint8) * 255

        # Save prediction
        save_path = os.path.join(output_dir, image_name)

        cv2.imwrite(save_path, prediction)

        # ==========================================================
        # Display Original and Prediction Side by Side
        # ==========================================================

        original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        original = cv2.resize(original, (IMAGE_SIZE, IMAGE_SIZE))

        display = np.hstack((original, prediction))

        cv2.imshow("Original (Left) | Prediction (Right)", display)

        key = cv2.waitKey(0)

        if key == 27:
            break