"""Formatting and export utilities for OCTA vessel biomarkers.

This module is intentionally limited to presentation and file export of an
already-computed biomarker dictionary. It does not compute biomarkers, inspect
graphs, or import graph-processing dependencies.
"""

from __future__ import annotations
import csv
import json
from pathlib import Path

import pandas as pd

__all__ = [
    "format_report",
    "print_report",
    "save_csv",
    "save_json",
    "save_excel",
]


def _validate_output_path(output_path: Path, expected_suffix: str) -> None:
    """Validate an output path and create missing parent directories.

    Args:
        output_path: Destination path for an exported report.
        expected_suffix: Required file suffix, including the leading dot.

    Raises:
        TypeError: If ``output_path`` is not a ``Path`` instance.
        ValueError: If ``expected_suffix`` is invalid or the path suffix does
            not match it.
        OSError: If parent directories cannot be created.
    """
    if not isinstance(output_path, Path):
        raise TypeError("output_path must be a pathlib.Path instance.")

    if not expected_suffix.startswith("."):
        raise ValueError("expected_suffix must include a leading dot.")

    if output_path.suffix.lower() != expected_suffix.lower():
        raise ValueError(
            f"Expected output file with suffix '{expected_suffix}', "
            f"got '{output_path.suffix or '<none>'}'."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)


def format_report(biomarkers: dict[str, float | int]) -> str:
    """Format computed OCTA vessel biomarkers as a readable text report.

    Args:
        biomarkers: Mapping returned by ``compute_biomarkers()``.

    Returns:
        A formatted multi-line biomarker report.

    Raises:
        TypeError: If ``biomarkers`` is not a dictionary.
    """
    if not isinstance(biomarkers, dict):
        raise TypeError("biomarkers must be a dictionary.")

    sections = [
        (
            "Length Metrics",
            [
                ("Total Vessel Length", "total_vessel_length"),
                ("Average Vessel Length", "average_vessel_length"),
                ("Average Edge Length", "average_edge_length"),
                ("Vessel Length Density", "vessel_length_density"),
            ],
        ),
        (
            "Topology Metrics",
            [
                ("Endpoint Count", "endpoint_count"),
                ("Branch Point Count", "branch_point_count"),
                ("Branch Point Density", "branch_point_density"),
                ("Isolated Nodes", "number_of_isolated_nodes"),
            ],
        ),
        (
            "Connectivity Metrics",
            [
                ("Connected Components", "connected_component_count"),
                (
                    "Largest Connected Component",
                    "largest_connected_component_size",
                ),
                ("Average Node Degree", "average_node_degree"),
                ("Graph Density", "graph_density"),
                ("Graph Transitivity", "graph_transitivity"),
                (
                    "Average Clustering Coefficient",
                    "average_clustering_coefficient",
                ),
            ],
        ),
        (
            "Tortuosity Metrics",
            [
                ("Average Tortuosity", "average_edge_tortuosity"),
                ("Maximum Tortuosity", "maximum_tortuosity"),
                ("Minimum Tortuosity", "minimum_tortuosity"),
                ("Tortuosity Std Dev", "tortuosity_standard_deviation"),
            ],
        ),
    ]

    separator = "=" * 40
    lines = [
        separator,
        "OCTA Vessel Biomarker Report",
        separator,
        "",
    ]

    for title, metrics in sections:
        lines.extend([title, "-" * len(title)])
        for label, key in metrics:
            value = biomarkers.get(key)

            if value is None:
                value = "N/A"
            elif isinstance(value, float):
                value = f"{value:.6f}"
            lines.append(f"{label:<34}: {value}")
        lines.append("")

    lines.append(separator)
    return "\n".join(lines)


def print_report(biomarkers: dict[str, float | int]) -> None:
    """Print a formatted OCTA vessel biomarker report.

    Args:
        biomarkers: Mapping returned by ``compute_biomarkers()``.
    """
    print(format_report(biomarkers))


def save_csv(biomarkers: dict[str, float | int], output_path: Path) -> None:
    """Save biomarkers to a two-column CSV file.

    Args:
        biomarkers: Mapping returned by ``compute_biomarkers()``.
        output_path: Destination ``.csv`` path.

    Raises:
        TypeError: If ``biomarkers`` is not a dictionary or ``output_path`` is
            not a ``Path`` instance.
        ValueError: If ``output_path`` does not end in ``.csv``.
        OSError: If the file cannot be written.
    """
    if not isinstance(biomarkers, dict):
        raise TypeError("biomarkers must be a dictionary.")

    _validate_output_path(output_path, ".csv")

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Metric", "Value"])
        writer.writerows(biomarkers.items())


def save_json(biomarkers: dict[str, float | int], output_path: Path) -> None:
    """Save biomarkers to a JSON file.

    Args:
        biomarkers: Mapping returned by ``compute_biomarkers()``.
        output_path: Destination ``.json`` path.

    Raises:
        TypeError: If ``biomarkers`` is not a dictionary or ``output_path`` is
            not a ``Path`` instance.
        ValueError: If ``output_path`` does not end in ``.json``.
        OSError: If the file cannot be written.
    """
    if not isinstance(biomarkers, dict):
        raise TypeError("biomarkers must be a dictionary.")

    _validate_output_path(output_path, ".json")

    with output_path.open("w", encoding="utf-8") as json_file:
        json.dump(biomarkers, json_file, indent=4)


def save_excel(biomarkers: dict[str, float | int], output_path: Path) -> None:
    """Save biomarkers to a single-worksheet Excel workbook.

    Args:
        biomarkers: Mapping returned by ``compute_biomarkers()``.
        output_path: Destination ``.xlsx`` path.

    Raises:
        TypeError: If ``biomarkers`` is not a dictionary or ``output_path`` is
            not a ``Path`` instance.
        ValueError: If ``output_path`` does not end in ``.xlsx``.
        OSError: If the file cannot be written.
    """
    if not isinstance(biomarkers, dict):
        raise TypeError("biomarkers must be a dictionary.")

    _validate_output_path(output_path, ".xlsx")

    dataframe = pd.DataFrame(
        biomarkers.items(),
        columns=["Metric", "Value"],
    )
    try:
        dataframe.to_excel(
            output_path,
            index=False,
            sheet_name="Biomarkers",
        )
    except ImportError as exc:
        raise ImportError(
            "Saving Excel reports requires an installed pandas Excel writer "
            "engine, such as openpyxl."
        ) from exc
