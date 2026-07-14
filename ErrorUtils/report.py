"""
==========================================================
Error Report Generator
==========================================================
"""

import os
import csv


def initialize_report(csv_path):
    """
    Create CSV file if it does not exist.
    """

    if not os.path.exists(csv_path):

        with open(csv_path, "w", newline="") as file:

            writer = csv.writer(file)

            writer.writerow([
                "Image",
                "X",
                "Y",
                "Detected Error"
            ])


def save_annotation(csv_path,
                    image_name,
                    x,
                    y,
                    error):

    """
    Append annotation information.
    """

    with open(csv_path, "a", newline="") as file:

        writer = csv.writer(file)

        writer.writerow([
            image_name,
            x,
            y,
            error
        ])