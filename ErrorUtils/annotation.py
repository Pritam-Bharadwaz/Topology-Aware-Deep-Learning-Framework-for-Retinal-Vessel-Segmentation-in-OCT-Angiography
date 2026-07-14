"""
==========================================================
Annotation Functions
==========================================================
"""

import cv2


COLORS = {
    "Linkage Error": (0, 0, 255),          # Red
    "Branching Error": (255, 0, 0),        # Blue
    "False Connection": (0, 255, 0),       # Green
    "Missing Vessel": (0, 255, 255),       # Yellow
    "False Vessel": (255, 0, 255)          # Purple
}


def draw_annotations(image, annotations):
    """
    Draw all saved annotations.
    """

    output = image.copy()

    for ann in annotations:

        x = ann["x"]
        y = ann["y"]
        label = ann["label"]

        color = COLORS[label]

        cv2.circle(output, (x, y), 18, color, 2)

        cv2.putText(
            output,
            label,
            (x + 20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2
        )

    return output