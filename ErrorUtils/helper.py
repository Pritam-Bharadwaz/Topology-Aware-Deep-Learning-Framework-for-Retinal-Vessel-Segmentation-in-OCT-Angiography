"""
==========================================================
Helper Functions
==========================================================
"""

import os
import cv2


def load_image(path):
    """
    Load an image in grayscale.
    """

    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(f"Cannot load image: {path}")

    return image


def extract_patch(image, x, y, patch_size=60):
    """
    Extract a square patch around the clicked point.
    """

    half = patch_size // 2

    x1 = max(0, x - half)
    y1 = max(0, y - half)

    x2 = min(image.shape[1], x + half)
    y2 = min(image.shape[0], y + half)

    patch = image[y1:y2, x1:x2]

    return patch