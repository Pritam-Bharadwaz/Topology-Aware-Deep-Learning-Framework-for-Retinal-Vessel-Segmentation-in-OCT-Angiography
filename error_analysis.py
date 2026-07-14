"""
Manual Error Annotation Tool
"""

import os
import cv2
from ErrorUtils.helper import load_image
from ErrorUtils.annotation import draw_annotations
from ErrorUtils.report import initialize_report, save_annotation

INPUT_FOLDER = "Dataset/OCTA-500/Selected_6"
OUTPUT_FOLDER = "Dataset/OCTA-500/error_output"
CSV_PATH = os.path.join(OUTPUT_FOLDER, "Error_Report.csv")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
initialize_report(CSV_PATH)

ERROR_TYPES = {
    ord("1"): "Linkage Error",
    ord("2"): "Branching Error",
    ord("3"): "False Connection",
    ord("4"): "Missing Vessel",
    ord("5"): "False Vessel",
}

current_label = "Linkage Error"
annotations = []
base_image = None
display_image = None

def refresh():
    global display_image
    display_image = draw_annotations(base_image, annotations)
    cv2.putText(display_image, f"Current: {current_label}", (10,25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.imshow("Topology Error Annotation", display_image)

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        annotations.append({"x":x, "y":y, "label":current_label})
        refresh()

image_list = sorted(os.listdir(INPUT_FOLDER))

for image_name in image_list:
    annotations.clear()
    gray = load_image(os.path.join(INPUT_FOLDER, image_name))
    base_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    cv2.namedWindow("Topology Error Annotation")
    cv2.setMouseCallback("Topology Error Annotation", mouse_callback)
    refresh()

    while True:
        key = cv2.waitKey(20) & 0xFF

        if key in ERROR_TYPES:
            current_label = ERROR_TYPES[key]
            refresh()

        elif key in (ord("u"), ord("U")):
            if annotations:
                annotations.pop()
            refresh()

        elif key in (ord("c"), ord("C")):
            annotations.clear()
            refresh()

        elif key in (ord("s"), ord("S")):
            out = draw_annotations(base_image, annotations)
            save_path = os.path.join(OUTPUT_FOLDER, image_name)
            cv2.imwrite(save_path, out)
            for ann in annotations:
                save_annotation(CSV_PATH, image_name, ann["x"], ann["y"], ann["label"])
            print("Saved:", save_path)

        elif key in (ord("n"), ord("N")):
            break

        elif key == 27:
            cv2.destroyAllWindows()
            raise SystemExit

cv2.destroyAllWindows()
print("Done")
