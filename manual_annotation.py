"""
==========================================================
Manual Error Annotation Tool
==========================================================
"""

import os
import cv2

INPUT_FOLDER = "Dataset/OCTA-500/Selected_6"
OUTPUT_FOLDER = "Dataset/OCTA-500/Error_Output"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

image_list = sorted(os.listdir(INPUT_FOLDER))

print(f"Images Found : {len(image_list)}")


# ==========================================================
# Mouse Callback
# ==========================================================

def draw_circle(event, x, y, flags, param):

    global image

    if event == cv2.EVENT_LBUTTONDOWN:

        cv2.circle(image, (x, y), 20, (0, 0, 255), 2)

        cv2.putText(
            image,
            "Linkage",
            (x + 25, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )

        cv2.imshow("Error Annotation Tool", image)



# ==========================================================
# Open Images
# ==========================================================

for image_name in image_list:

    image_path = os.path.join(INPUT_FOLDER, image_name)

    image = cv2.imread(image_path)

    if image is None:
        print(f"Cannot open {image_name}")
        continue

    cv2.imshow("Error Annotation Tool", image)

    cv2.setMouseCallback(
        "Error Annotation Tool",
        draw_circle
    )

    key = cv2.waitKey(0)

    cv2.destroyAllWindows()