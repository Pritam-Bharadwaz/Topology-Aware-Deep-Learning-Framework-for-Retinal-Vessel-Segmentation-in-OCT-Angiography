import os
import cv2
import numpy as np
from skimage.morphology import skeletonize

from config import (
    TRAIN_LABEL_DIR,
    TRAIN_SKELETON_DIR,
    VAL_LABEL_DIR,
    VAL_SKELETON_DIR,
    TEST_LABEL_DIR,
    TEST_SKELETON_DIR,
)


def generate_skeletons(label_dir, skeleton_dir):

    os.makedirs(skeleton_dir, exist_ok=True)

    files = sorted(os.listdir(label_dir))

    count = 0

    for file in files:

        if not file.endswith(".bmp"):
            continue

        label_path = os.path.join(label_dir, file)

        mask = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        if mask is None:
            print(f"Could not read {file}")
            continue

        # Convert to binary
        binary = mask > 127

        # Skeletonize
        skeleton = skeletonize(binary)

        # Convert back to 0–255 image
        skeleton = (skeleton.astype(np.uint8)) * 255

        save_path = os.path.join(skeleton_dir, file)

        cv2.imwrite(save_path, skeleton)

        count += 1

    print(f"{count} skeletons saved in {skeleton_dir}")


if __name__ == "__main__":

    print("Generating Training Skeletons...")
    generate_skeletons(TRAIN_LABEL_DIR, TRAIN_SKELETON_DIR)

    print("Generating Validation Skeletons...")
    generate_skeletons(VAL_LABEL_DIR, VAL_SKELETON_DIR)

    print("Generating Testing Skeletons...")
    generate_skeletons(TEST_LABEL_DIR, TEST_SKELETON_DIR)

    print("\nAll skeleton images generated successfully!")